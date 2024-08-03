import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm

from dataloaders.dataloader import DataLoaderCustom
from networks.NewCRFDepth import NewCRFDepth
from parser_options import eval_parser
from custom_logging import tb_visualization, tb_visualization_unc
from utils import compute_errors, sigma_metric_from_canonical_and_scale, sigma_metric_from_canonical_and_scale_nddepth
from aucs import compute_aucs, SCC

# Parse config file #
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
else:
    args = eval_parser.parse_args()


def eval_func(model, dataloader_eval):
    eval_measures = torch.zeros(11).cuda()
    eval_measures_percentage = torch.zeros(2).cuda()

    uncertainty_metrics = ["abs_rel", "rmse", "a1"]
    aucs = {"abs_rel": [], "rmse": [], "a1": []}    # Save results 
    scc_total = 0                                   # Spearman correlation coefficient in L1 score

    num_semantic_classes = dataloader_eval.num_semantic_classes

    sigma_metric_from_canonical_and_scale_func = sigma_metric_from_canonical_and_scale if args.unc_loss_type != 2 else sigma_metric_from_canonical_and_scale_nddepth
    
    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)

    excecution_times = []
    for step, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            # Init #
            pred_depths_r_list, pred_depths_u_list =  [], []
            
            segmentation_map, instances, labels = None, None, None

            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']

            if args.instances: 
                instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda())
                boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda())
                labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda())
            if args.segmentation:
                segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda())

            # Predict #
            start_time = time.time()
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image)
            end_time = time.time()
            excecution_times.append(end_time - start_time)

            # Unpack result #
            pred_depths_r_list = result["pred_depths_r_list"]

            # Depth #
            if args.instances:
                pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
            elif args.segmentation:
                pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
            else:  # Global
                pred_depth = pred_depths_r_list[-1]

            # Uncertainty #
            if args.eval_unc:
                if args.unc_head:
                    sigma_m = sigma_metric_from_canonical_and_scale_func(
                        result['pred_depths_rc_list'][-1], result["unc_c"][-1],
                        result['pred_scale_list'][-1].unsqueeze(-1).unsqueeze(-1),
                        result["unc_s"][-1].unsqueeze(-1).unsqueeze(-1), args)
                # Uncertainty of bins is std --> convert to variance #
                else:
                    sigma_m = sigma_metric_from_canonical_and_scale_func(
                        result['pred_depths_rc_list'][-1], result['uncertainty_maps_list'][-1]**2,
                        result['pred_scale_list'][-1].unsqueeze(-1).unsqueeze(-1),
                        (result["uncertainty_maps_scale_list"][-1]**2), args)
                # Mask #
                if args.instances:
                    sigma_m = torch.sum((sigma_m * instances), dim=1).unsqueeze(1)
                elif args.segmentation:
                    sigma_m = torch.sum((sigma_m * segmentation_map), dim=1).unsqueeze(1)
                else:
                    pass

            # Tensorboard #
            if args.eval_unc:
                tb_visualization_unc(writer, None, sigma_m, global_step=step, args=args, current_lr=None, \
                             var_sum=None, var_cnt=None, num_images=1, depth_gt=gt_depth, image=image, max_tree_depth=args.max_tree_depth, \
                             pred_depths_r_list=pred_depths_r_list, pred_depths_rc_list=result["pred_depths_rc_list"], \
                             num_semantic_classes=num_semantic_classes, instances=instances, segmentation_map=segmentation_map, \
                             labels=labels, pred_depths_c_list=result["pred_depths_c_list"], uncertainty_maps_list=result["uncertainty_maps_list"], \
                             pred_depths_u_list=pred_depths_u_list, expensive_viz=True)
            else:
                tb_visualization(writer=writer, global_step=step, args=args, current_loss_depth=None, current_lr=None, \
                             var_sum=None, var_cnt=None, num_images=1, depth_gt=gt_depth, image=image, max_tree_depth=args.max_tree_depth, \
                             pred_depths_r_list=pred_depths_r_list, pred_depths_rc_list=result["pred_depths_rc_list"], \
                             num_semantic_classes=num_semantic_classes, instances=instances, segmentation_map=segmentation_map, \
                             labels=labels, pred_depths_c_list=result["pred_depths_c_list"], uncertainty_maps_list=result["uncertainty_maps_list"], \
                             pred_depths_u_list=pred_depths_u_list, expensive_viz=True)

            # To numpy #
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
            if args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1
            else:
                eval_mask = gt_depth > 0.1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            if np.all(gt_depth[valid_mask] == 0) or len(gt_depth[valid_mask]) == 1:
                continue

        # Uncertainty eval #
        unc_error = None

        if args.eval_unc:
            # Computer also error^2/var #
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])
            eval_measures[10] += torch.tensor(measures[-1]).cuda()

            # AUCS + spearman_coefficient # 
            scores = compute_aucs(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])
            [aucs[m].append(scores[m]) for m in uncertainty_metrics]

            scc_total += SCC(gt_depth[valid_mask], pred_depth[valid_mask], sigma_m[valid_mask])
        else:
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        # Some basic depth metrics #
        eval_measures[:9] += torch.tensor(measures[:9]).cuda() 
        eval_measures[9] += 1 # Size of dataset 
   
    # Depth related + optional e^2/variance #
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

    excecution_times = np.asarray(excecution_times)
    print(f'Excecutation time : {excecution_times.mean()} seconds')

    return eval_measures_cpu, unc_error


def main_worker(args):

    dataloader_eval = DataLoaderCustom(args, 'eval')
    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_instances = dataloader_eval.num_instances
    
    # Model #
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
        eval_measures, eval_measures_unc = eval_func(model, dataloader_eval)


def main():
    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
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
