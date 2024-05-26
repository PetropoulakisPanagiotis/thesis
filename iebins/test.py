import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn

import cv2
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import argparse
import numpy as np
import random
from tqdm import tqdm

from dataloaders.dataloader import DataLoaderCustom
from networks.NewCRFDepth import NewCRFDepth
from parser_options import convert_arg_line_to_args, test_parser
from custom_logging import debug_result, debug_visualize_gt_instances, tb_visualization, tb_visualization_d3vo
from utils import compute_errors, sigma_metric_from_canonical_and_scale, colormap, map_float_data_to_int
from aucs import compute_aucs,  SCC

# Parse config file #
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = test_parser.parse_args([arg_filename_with_prefix])
elif sys.argv.__len__() == 3:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
    args.pick_class = torch.tensor(int(sys.argv[2]))
else:
    args = eval_parser.parse_args()


def predict(model, dataloader_eval) -> None:
    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances

    with open(args.filenames_file_eval, 'r') as f:
        file_id_str = f.read().splitlines()

    for step, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            # Init 
            pred_depths_r_list, pred_depths_rc_list, pred_depths_instances_r_list, \
                pred_depths_instances_rc_list, pred_depths_c_list, uncertainty_maps_list, \
                pred_depths_u_list = [], [], [], [], [], [], []
            segmentation_map, instances, labels, unc_decoder = None, None, None, None

            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']

            if (args.dataset == 'nyu' or args.dataset == 'scannet') and args.segmentation:
                segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda())
                instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda())
                boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda())
                labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda())
            
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            # Predict
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image)
            
            # Unpack result         
            if args.instances:
                pred_depths_r_list = result["pred_depths_instances_r_list"]

                pred_depths_instances_rc_list = result["pred_depths_instances_rc_list"]
                pred_depths_instances_r_list = result["pred_depths_instances_r_list"]
            else:
                pred_depths_r_list = result["pred_depths_r_list"]
                # Canonical segmentation/single scale #
                if args.update_block != 0 and args.update_block != 4 and args.update_block != 3:    
                    pred_depths_rc_list = result["pred_depths_rc_list"]
            
            # Uncertainty bins #
            if args.update_block == 0 or args.update_block == 3 or args.update_block == 18 \
               or args.update_block == 1 or args.update_block == 2:
                pred_depths_c_list = result["pred_depths_c_list"]
                uncertainty_maps_list = result["uncertainty_maps_list"]
           
            if args.instances:
                # Fair comparison segmentation - instances: use eval_per_class.sh script   
                if args.pick_class != 0: 
                    non_class = torch.nonzero(labels[0] != args.pick_class)
                    if non_class.shape[0] == 63:
                        continue
                    instances[0, non_class] = torch.zeros_like(instances[0, non_class])
                
                pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
                
                mask = torch.sum(instances, dim=1).unsqueeze(0).to(torch.bool).cpu()
                gt_depth = (gt_depth * mask)
                #wals = torch.nonzero(labels[0] == 1)
                #print(result['pred_shift_instances_list'][-1][0, wals])
                #print(result['pred_scale_instances_list'][-1][0, wals])
            elif args.segmentation:

                pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)

                if args.d3vo:
                    sigma_c = torch.sum((segmentation_map * (result["uncertainty_maps_list"][-1] **2)), dim=1)
                    c = torch.sum((segmentation_map * result["pred_depths_rc_list"][-1]), dim=1)
                    
                    if args.d3vo_c:
                        sigma_s = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1], result["unc_d3vo_c"], result['pred_scale_list'][-1], result["unc_d3vo"], args)
                        sigma_s = torch.sum((sigma_s * segmentation_map), dim=1).unsqueeze(1)
                    else:
                        # uncertainty of canonical is std --> convert to variance
                        sigma_s = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1], result['uncertainty_maps_list'][-1] ** 2, result['pred_scale_list'][-1], result["unc_d3vo"], args)
                        sigma_s = torch.sum((sigma_s * segmentation_map), dim=1).unsqueeze(1)

                    s = torch.sum((segmentation_map * result["pred_scale_list"][-1].unsqueeze(-1).unsqueeze(-1)), dim=1)
                    sigma_m = s**2 * sigma_c + c**2 * sigma_s
                    sigma_m = sigma_m.unsqueeze(0)
                # Fair comparison segmentation - instances: use eval_per_class.sh script   
                if True:
                    if args.pick_class != 0:    
                        non_class = torch.nonzero(labels[0] != args.pick_class)
                        if non_class.shape[0] == 63:
                            continue
                        instances[0, non_class] = torch.zeros_like(instances[0, non_class])
                    
                    if False:
                        tmp = instances_mask.permute(1,2,0)
                        cv2.imshow("instances_mapped_image", (tmp.cpu().numpy() * 255).astype('uint8'))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
                    
                    mask = torch.sum(instances, dim=1).unsqueeze(0)
                    pred_depth = pred_depth * mask

                    mask = mask.to(torch.bool).cpu()
                    gt_depth = (gt_depth * mask)
            else:
                pred_depth = pred_depths_r_list[-1]

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()     
            if args.d3vo:
                sigma_m = sigma_m.cpu().numpy().squeeze()
        
        pred_depth[pred_depth < args.min_depth_test] = args.min_depth_test
        pred_depth[pred_depth > args.max_depth_test] = args.max_depth_test
        pred_depth[np.isinf(pred_depth)] = args.max_depth_test
        pred_depth[np.isnan(pred_depth)] = args.min_depth_test

        # Save depth metric #
        filename = file_id_str[step].split('/')[1] + '.png'
        normalization_const = 1e3

        cv2.imwrite(args.save_dir + 'depth/' + filename, map_float_data_to_int(pred_depth, normalization_const), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if args.segmentation:
            pass
        elif args.instances:
            pass
        else:
            # Scale #
            scale = result['pred_scale_list'][-1].cpu().numpy().squeeze()
            scale_map = np.ones_like(pred_depth) * scale
            
            cv2.imwrite(args.save_dir + 'scale/' + filename, map_float_data_to_int(scale_map, normalization_const), [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Canonical #    
            canonical = result['pred_depths_rc_list'][-1].cpu().numpy().squeeze()
            cv2.imwrite(args.save_dir + 'canonical/' + filename, map_float_data_to_int(canonical, normalization_const), [cv2.IMWRITE_PNG_COMPRESSION, 9])    

        if False:
            gt_depth[gt_depth < args.min_depth_test] = args.min_depth_test
            gt_depth[gt_depth > args.max_depth_test] = args.max_depth_test
            gt_depth[np.isinf(gt_depth)] = args.max_depth_test
            gt_depth[np.isnan(gt_depth)] = args.min_depth_test
            valid_mask = np.logical_and(gt_depth > args.min_depth_test, gt_depth < args.max_depth_test)
            cv2.imshow('i', colormap(np.log(pred_depth[np.newaxis,:]), name='magma').transpose(1, 2, 0))
            cv2.imshow('i', colormap(np.log(gt_depth[np.newaxis,:]), name='magma').transpose(1, 2, 0))
            cv2.waitKey(0)       
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
    

def main_worker(args):
    
    dataloader_eval = DataLoaderCustom(args, 'online_eval')

    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances

    # Depth model
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type, 
                        num_semantic_classes=num_semantic_classes, num_instances=num_instances, var=args.var, \
                        padding_instances=args.padding_instances, \
                        segmentation_active=args.segmentation,  instances_active=args.instances,\
                        roi_align=args.roi_align, roi_align_size=args.roi_align_size, \
                        bins_scale=args.bins_scale, d3vo=args.d3vo, d3vo_c=args.d3vo_c, virtual_depth_variation=args.virtual_depth_variation)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))
    model = torch.nn.DataParallel(model)
    model.cuda()
    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
            exit()

    cudnn.benchmark = True

    # Evaluate #
    model.eval()
    with torch.no_grad():
        predict(model, dataloader_eval)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()

    print('Save dir: ', args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + 'depth/', exist_ok=True)
    os.makedirs(args.save_dir + 'scale/', exist_ok=True)
    os.makedirs(args.save_dir + 'canonical/', exist_ok=True)
    os.makedirs(args.save_dir + 'uncertainty/', exist_ok=True)

    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)

if __name__ == '__main__':
    main()
