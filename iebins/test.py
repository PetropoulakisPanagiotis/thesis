import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn

import cv2

import numpy as np
from tqdm import tqdm

from dataloaders.dataloader import DataLoaderCustom
from networks.NewCRFDepth import NewCRFDepth
from parser_options import test_parser
from utils import compute_errors, colormap, map_float_data_to_int

# Parse config file #
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = test_parser.parse_args([arg_filename_with_prefix])
else:
    args = test_parser.parse_args()


def predict(model, dataloader_eval) -> None:
    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances

    # Open test files #
    with open(args.filenames_file_eval, 'r') as f:
        file_id_str = f.read().splitlines()

    for step, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
        gt_depth = eval_sample_batched['depth']

        if args.instances:
            instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda())
            boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda())
            labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda())
        
        if args.segmentation:
            segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda())

        # Predict
        if args.instances:
            result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
        elif args.segmentation:
            result = model(image, masks=segmentation_map)
        else:
            result = model(image)

        # Unpack result #
        pred_depths_r_list = result["pred_depths_r_list"]

        # Depth prediction #
        if args.instances:
            pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
        elif args.segmentation:
            pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
        else:
            pred_depth = pred_depths_r_list[-1]

        pred_depth = pred_depth.cpu().numpy().squeeze()
        gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_test] = args.min_depth_test
        pred_depth[pred_depth > args.max_depth_test] = args.max_depth_test
        pred_depth[np.isinf(pred_depth)] = args.max_depth_test
        pred_depth[np.isnan(pred_depth)] = args.min_depth_test

        # Save depth metric #
        filename_base = file_id_str[step].split('/')[1]
        filename = filename_base + '.png'
        normalization_const_depth = 1e3

        cv2.imwrite(args.save_dir + 'depth/' + filename, map_float_data_to_int(pred_depth, normalization_const_depth),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        if args.instances:
            # Scale #
            scale = result['pred_scale_list'][-1].cpu().numpy()[0].tolist()
            scale_data = {'scale': scale, 'scale_type': 'per-instance'}
            scale_idx = [i for i in range(len(scale))]

            # Scale_map: h,w -> scale id
            scale_idx_tensor = torch.tensor(
                scale_idx, dtype=torch.int16).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device=instances.device)
            scale_map = torch.sum((instances * scale_idx_tensor), dim=1).to(torch.int16).squeeze(0)
            cv2.imwrite(args.save_dir + 'scale_map/' + filename,
                        scale_map.cpu().numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Which network scale predictions are valid #
            valid_scales = torch.unique(scale_map).cpu().numpy().tolist()
            scale_data['valid'] = [1 if scale_idx_item in valid_scales else 0 for scale_idx_item in scale_idx]

            # Canonical #
            canonical = torch.sum((instances * result["pred_depths_rc_list"][-1]), dim=1)
            canonical = canonical.cpu().numpy().squeeze()

            cv2.imwrite(args.save_dir + 'canonical/' + filename,
                        map_float_data_to_int(canonical, normalization_const_depth), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            # Uncertainty canonical and scale #
            if args.unc_head:
                canonical_unc = result["unc_c"][-1]
                scale_unc = result["unc_s"][-1]
            elif args.virtual_depth_variation == 0:
                canonical_unc = result["uncertainty_maps_list"][-1]**2  # to variance
                canonical_unc = torch.sum((instances * canonical_unc), dim=1).to(torch.int16).squeeze(0)
                
                scale_unc = result["uncertainty_maps_scale_list"][-1]**2  # to variance
            else:
                raise ValueError(
                    "Can not estimate uncertainty. Either train a model with extra uncertainty heads or with bins for both scale/canonical\n"
                )

            scale_unc = scale_unc.view(num_instances).cpu().numpy().tolist()
            scale_data['scale_uncertainty'] = scale_unc

            canonical_unc = canonical_unc.cpu().numpy()
            np.save(args.save_dir + 'canonical_unc/' + filename_base + ".npy", canonical_unc)
            with open(args.save_dir + 'scale/' + filename_base + ".json", 'w') as file:
                json.dump(scale_data, file, indent=4)

        elif args.segmentation:
            # Scale #
            scale = result['pred_scale_list'][-1].cpu().numpy()[0].tolist()
            scale_data = {'scale': scale, 'scale_type': 'per-class'}
            scale_idx = [i for i in range(len(scale))]

            scale_idx_tensor = torch.tensor(
                scale_idx,
                dtype=torch.int16).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device=segmentation_map.device)
            
            scale_map = torch.sum((segmentation_map * scale_idx_tensor), dim=1).to(torch.int16).squeeze(0)
            cv2.imwrite(args.save_dir + 'scale_map/' + filename,
                        scale_map.cpu().numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 9])

            valid_scales = torch.unique(scale_map).cpu().numpy().tolist()
            scale_data['valid'] = [1 if scale_idx_item in valid_scales else 0 for scale_idx_item in scale_idx]

            # Canonical #
            canonical = torch.sum((segmentation_map * result["pred_depths_rc_list"][-1]), dim=1)
            canonical = canonical.cpu().numpy().squeeze()

            cv2.imwrite(args.save_dir + 'canonical/' + filename,
                        map_float_data_to_int(canonical, normalization_const_depth), [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Uncertainty canonical and scale #
            if args.unc_head:
                canonical_unc = result["unc_c"][-1]
                scale_unc = result["unc_s"][-1]
            elif args.virtual_depth_variation == 0:
                canonical_unc = result["uncertainty_maps_list"][-1]**2  # to variance
                canonical_unc = torch.sum((segmentation_map * canonical_unc), dim=1).to(torch.int16).squeeze(0)

                scale_unc = result["uncertainty_maps_scale_list"][-1]**2  # to variance
            else:
                raise ValueError(
                    "Can not estimate uncertainty. Either train a model with extra uncertainty heads or with bins for both scale/canonical\n"
                )
            scale_unc = scale_unc.view(num_semantic_classes).cpu().numpy().tolist()
            scale_data['scale_uncertainty'] = scale_unc

            canonical_unc = canonical_unc.cpu().numpy()
            np.save(args.save_dir + 'canonical_unc/' + filename_base + ".npy", canonical_unc)

            with open(args.save_dir + 'scale/' + filename_base + ".json", 'w') as file:
                json.dump(scale_data, file, indent=4)

        # Global scale #
        else:  
            # Scale #
            scale = float(result['pred_scale_list'][-1].cpu().numpy().squeeze())
            scale_data = {'scale': [scale], 'scale_type': 'global', 'valid': [1]}

            # Canonical #
            canonical = result['pred_depths_rc_list'][-1].cpu().numpy().squeeze()
            cv2.imwrite(args.save_dir + 'canonical/' + filename,
                        map_float_data_to_int(canonical, normalization_const_depth), [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Ids of scales -> one scale, hence 0 idx #
            cv2.imwrite(args.save_dir + 'scale_map/' + filename, np.zeros_like(canonical),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Uncertainty canonical and scale #
            if args.unc_head:
                canonical_unc = result["unc_c"][-1]
                scale_unc = result["unc_s"][-1]
            elif args.virtual_depth_variation == 0:
                canonical_unc = result["uncertainty_maps_list"][-1]**2  # to variance
                scale_unc = result["uncertainty_maps_scale_list"][-1]**2  # to variance
            else:
                raise ValueError(
                    "Can not estimate uncertainty. Either train a model with extra uncertainty heads or with bins for both scale/canonical\n"
                )
            scale_unc = scale_unc.squeeze(0).cpu().numpy().tolist()
            scale_data["scale_uncertainty"] = scale_unc

            canonical_unc = canonical_unc.cpu().numpy()

            np.save(args.save_dir + 'canonical_unc/' + filename_base + ".npy", canonical_unc)

            with open(args.save_dir + 'scale/' + filename_base + ".json", 'w') as file:
                json.dump(scale_data, file, indent=4)
        
        debug = False
        if debug:
            gt_depth[gt_depth < args.min_depth_test] = args.min_depth_test
            gt_depth[gt_depth > args.max_depth_test] = args.max_depth_test
            gt_depth[np.isinf(gt_depth)] = args.max_depth_test
            gt_depth[np.isnan(gt_depth)] = args.min_depth_test
            cv2.imshow('i', colormap(np.log(pred_depth[np.newaxis, :]), name='magma').transpose(1, 2, 0))
            cv2.imshow('i', colormap(np.log(gt_depth[np.newaxis, :]), name='magma').transpose(1, 2, 0))
            cv2.waitKey(0)
            valid_mask = np.logical_and(gt_depth > args.min_depth_test, gt_depth < args.max_depth_test)
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            print(measures)

def main_worker(args):
    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)

    dataloader_eval = DataLoaderCustom(args, 'eval')

    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances

    # Depth model #
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type,
                        num_semantic_classes=num_semantic_classes, num_instances=num_instances, \
                        padding_instances=args.padding_instances, \
                        segmentation_active=args.segmentation, concat_masks=args.concat_masks, instances_active=args.instances,\
                        roi_align=args.roi_align, roi_align_size=args.roi_align_size, \
                        bins_scale=args.bins_scale, unc_head=args.unc_head, virtual_depth_variation=args.virtual_depth_variation, \
                        upsample_type=args.upsample_type, bins_type=args.bins_type, bins_type_scale=args.bins_type_scale)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))
    model = torch.nn.DataParallel(model)
    model.cuda()
    print("== Model Initialized")

    if args.checkpoint_path != '' and os.path.isfile(args.checkpoint_path):
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

    init_save_dir = args.save_dir 
    init_filenames_file_eval = args.filenames_file_eval
    if '.txt' not in args.filenames_file_eval:

        scenes = [f for f in os.listdir(args.filenames_file_eval) if f.startswith('scene')]

        for scene in tqdm(scenes, total=len(scenes)):
            args.save_dir = init_save_dir + scene.split('.')[0] + '/'

            args.filenames_file_eval = init_filenames_file_eval + '/' + scene

            if args.instances:
                args.save_dir += 'per_instance/'
            elif args.segmentation:
                args.save_dir += 'per_class/'
            else:
                args.save_dir += 'global/'

            print('Save dir: ', args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + 'depth/', exist_ok=True)
            os.makedirs(args.save_dir + 'scale/', exist_ok=True)
            os.makedirs(args.save_dir + 'scale_map/', exist_ok=True)
            os.makedirs(args.save_dir + 'canonical/', exist_ok=True)
            os.makedirs(args.save_dir + 'canonical_unc/', exist_ok=True)

            if ngpus_per_node > 1:
                print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
                return -1
            
            main_worker(args)

    else:
        args.save_dir += args.filenames_file_eval.split('/')[-1].split('.')[0] + '/'

        if args.instances:
            args.save_dir += 'instances/'
        elif args.segmentation:
            args.save_dir += 'segmentation/'
        else:
            args.save_dir += 'single/'

        print('Save dir: ', args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + 'depth/', exist_ok=True)
        os.makedirs(args.save_dir + 'scale/', exist_ok=True)
        os.makedirs(args.save_dir + 'scale_map/', exist_ok=True)
        os.makedirs(args.save_dir + 'canonical/', exist_ok=True)
        os.makedirs(args.save_dir + 'canonical_unc/', exist_ok=True)

        if ngpus_per_node > 1:
            print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
            return -1
        
        main_worker(args)



if __name__ == '__main__':
    main()
