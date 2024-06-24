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
from parser_options import convert_arg_line_to_args, eval_parser
from custom_logging import debug_result, debug_visualize_gt_instances, tb_visualization, tb_visualization_d3vo
from utils import compute_errors, sigma_metric_from_canonical_and_scale
from aucs import compute_aucs, SCC

# Parse config file #
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
elif sys.argv.__len__() == 3:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
    args.pick_class = torch.tensor(int(sys.argv[2]))
else:
    args = eval_parser.parse_args()


def eval_func(model, dataloader_eval):
    eval_measures = torch.zeros(11).cuda()

    uncertainty_metrics = ["abs_rel", "rmse", "a1"]
    aucs = {"abs_rel": [], "rmse": [], "a1": []}
    scc_total = 0  # Spearman correlation coefficient in l1

    # Uncertainty for metric depth from the decoder #
    eval_measures_unc = torch.zeros(1).cuda()

    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances

    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)

    scales = []
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
                for scale in result['pred_scale_instances_list'][-1]:
                    for scale_s in scale:
                        scales.append(float(scale_s.cpu().numpy()))
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
                for scale in result['pred_scale_list'][-1]:
                    for scale_s in scale:
                        scales.append(float(scale_s.cpu().numpy()))
            else:
                result = model(image)
                if args.update_block != 0:
                    for scale in result['pred_scale_list'][-1]:
                        scales.append(float(scale.cpu().numpy()))

            if False:
                debug_result(result, gt_depth)

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
            if args.update_block == 0 or args.update_block == 3 or args.update_block == 1 \
               or args.update_block == 18 or args.update_block == 2:
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
            elif args.segmentation:

                pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)

                if args.eval_unc:
                    if args.unc_head:
                        sigma_c = torch.sum((segmentation_map * (result["uncertainty_maps_list"][-1]**2)), dim=1)
                        c = torch.sum((segmentation_map * result["pred_depths_rc_list"][-1]), dim=1)

                        sigma_s = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1],
                                                                        result["unc_d3vo_c"],
                                                                        result['pred_scale_list'][-1],
                                                                        result["unc_d3vo"], args)
                        sigma_s = torch.sum((sigma_s * segmentation_map), dim=1).unsqueeze(1)
                    else:
                        # uncertainty of canonical is std --> convert to variance
                        sigma_s = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1],
                                                                        result['uncertainty_maps_list'][-1]**2,
                                                                        result['pred_scale_list'][-1],
                                                                        result["unc_d3vo"], args)
                        sigma_s = torch.sum((sigma_s * segmentation_map), dim=1).unsqueeze(1)

                    s = torch.sum((segmentation_map * result["pred_scale_list"][-1].unsqueeze(-1).unsqueeze(-1)), dim=1)
                    sigma_m = s**2 * sigma_c + c**2 * sigma_s
                    sigma_m = sigma_m.unsqueeze(0)
            else: # Global
                pred_depth = pred_depths_r_list[-1]

                if args.eval_unc:
                    if args.unc_head:
                        sigma_m = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1],
                                                                        result["unc_c"][-1],
                                                                        result['pred_scale_list'][-1].unsqueeze(-1).unsqueeze(-1),
                                                                        result["unc_s"][-1].unsqueeze(-1).unsqueeze(-1), args)
                    else: # Convert std to variance
                        sigma_m = sigma_metric_from_canonical_and_scale(result['pred_depths_rc_list'][-1],
                                                                    result['uncertainty_maps_list'][-1]**2,
                                                                    result['pred_scale_list'][-1].unsqueeze(-1).unsqueeze(-1),
                                                                    (result["uncertainty_maps_scale_list"][-1]**2).unsqueeze(-1).unsqueeze(-1), args)
            # Tensorboard
            if args.eval_unc:
                tb_visualization_d3vo(writer, global_step=step, args=args, current_loss_d3vo=None, current_lr=None, var_sum=None, var_cnt=None, \
                                              num_images=1, sigma_metric=sigma_m)
            else:
                tb_visualization(writer=writer, global_step=step, args=args, current_loss_depth=None, current_lr=None, current_loss_unc_decoder=None, \
                             var_sum=None, var_cnt=None, num_images=1, depth_gt=gt_depth, image=image, max_tree_depth=args.max_tree_depth, \
                             pred_depths_r_list=pred_depths_r_list, pred_depths_rc_list=pred_depths_rc_list, \
                             pred_depths_instances_r_list=pred_depths_instances_r_list, pred_depths_instances_rc_list=pred_depths_instances_rc_list, \
                             num_semantic_classes=num_semantic_classes, instances=instances, segmentation_map=segmentation_map, \
                             labels=labels, pred_depths_c_list=pred_depths_c_list, uncertainty_maps_list=uncertainty_maps_list, \
                             pred_depths_u_list=pred_depths_u_list, unc_decoder=None, expensive_viz=True)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            if args.eval_unc:
                sigma_m = sigma_m.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.dataset == 'scannet':
                eval_mask = gt_depth > 0.1

            if args.eigen_crop:
                if args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            if np.all(gt_depth[valid_mask] == 0) or len(gt_depth[valid_mask]) == 1:
                continue

        # For uncertainty #
        unc_error = None

        if args.eval_unc:
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])
            eval_measures[10] += torch.tensor(measures[-1]).cuda()

            scores = compute_aucs(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])
            [aucs[m].append(scores[m]) for m in uncertainty_metrics]

            scc_total += SCC(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])

        else:
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures[:9]).cuda()
        eval_measures[9] += 1

    if False:
        scales = np.array(scales)
        print("Mean scale: ", scales.mean())
        print("Std scale: ", scales.std())
        print("Max scale: ", scales.max())
        print("Min scale: ", scales.min())
    scales = np.array(scales)
    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))

    if args.eval_unc:

        for m in uncertainty_metrics:
            aucs[m] = np.array(aucs[m]).mean(0)

        print("\n  " + ("{:>8} | " * 6).format("abs_rel", "", "rmse", "", "a1", ""))
        print("  " + ("{:>8} | " * 6).format("AUSE", "AURG", "AUSE", "AURG", "AUSE", "AURG"))
        print(("&{:8.3f}  " * 6).format(*aucs["abs_rel"].tolist() + aucs["rmse"].tolist() + aucs["a1"].tolist()) +
              "\\\\")

        print("e_metric^2/variance: ", eval_measures[10].cpu().numpy() / cnt)
        print("SCC: ", scc_total / cnt)

    return eval_measures_cpu, unc_error


def main_worker(args):

    dataloader_eval = DataLoaderCustom(args, 'online_eval')
    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances
    # Depth model
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type,
                        num_semantic_classes=num_semantic_classes, num_instances=num_instances, \
                        padding_instances=args.padding_instances, \
                        segmentation_active=args.segmentation,  instances_active=args.instances,\
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
        eval_measures, eval_measures_unc = eval_func(model, dataloader_eval)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    exp_name = args.exp_name

    check_args(args)
    
    args.log_directory = os.path.join(args.log_directory, exp_name)

    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)
    print(args.log_directory)

    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    main_worker(args)


def check_args(args):
    if args.eval_unc:
        if args.update_block == 1 and args.virtual_depth_variation != 0 and not args.unc_head:
            print("Can not evaluate evalutate uncertainty. Please enable bins or predict uncertainty via extra heads\n")

    

if __name__ == '__main__':
    main()
