from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist

from networks.NewCRFDepth_clean import NewCRFDepth
from utils_clean import flip_lr, compute_errors, compute_error_uncertainty


def online_eval(args, model, dataloader_eval, gpu, epoch, ngpus, group, post_process=False):
    with torch.no_grad():
        # Save results #
        measures_size = 10
        eval_measures = torch.zeros(measures_size).cuda(device=gpu)
        
        # Uncertainty #
        eval_measures_unc = torch.zeros(1).cuda(device=gpu)

        for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            
            if args.segmentation:
                segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda(args.gpu, non_blocking=True))

            if args.instances:
                instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda(args.gpu, non_blocking=True))
                boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda(args.gpu, non_blocking=True))
                labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda(args.gpu, non_blocking=True))

            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            # Predict #
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image)

            if args.instances:
                pred_depths_r_list = result["pred_depths_instances_r_list"] 
            else:
                pred_depths_r_list = result["pred_depths_r_list"]
            
            # Uncertainty from decoder #
            if args.predict_unc:
                unc_decoder = result["unc_decoder"].cpu().numpy().squeeze()

            # Mask predictions #
            if args.instances:
                #instances[:, 6:, :, :] = 0
                pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
            elif args.segmentation:
                pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
            else:
                pred_depth = pred_depths_r_list[-1]

            pred_depth = pred_depth.cpu().numpy().squeeze()
            
            """
            #  Predict depth of flipped image and fuse
            if post_process:
                image_flipped = flip_lr(image)
                segmentation_map_flipped = flip_lr(segmentation_map)
                if args.update_block >= 9:
                    result = model(image_flipped, masks = segmentation_map_flipped)
                else:
                    result = model(image_flipped)
                pred_depths_r_list_flipped = result["pred_depths_r_list"]

                if args.segmentation:
                    pred_depth = post_process_depth(torch.sum((pred_depths_r_list[-1] * segmentation_map_flipped), dim=1).unsqueeze(0), torch.sum((pred_depths_r_list_flipped[-1] * segmentation_map), dim=1).unsqueeze(0))
                else:
                    pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])

                pred_depth = pred_depth.cpu().numpy().squeeze()
            """

            # Mask gt_depth #
            if args.instances:
                mask = torch.sum(instances, dim=1).unsqueeze(-1).to(torch.bool).cpu()
                gt_depth = (gt_depth * mask).cpu().numpy().squeeze()
            else:
                gt_depth = gt_depth.cpu().numpy().squeeze()

            if args.do_kb_crop:
                height, width = gt_depth.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped

            # Filter predicted depth #
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            # Final mask to evaluate #
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), \
                              int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), \
                                  int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    elif args.dataset == 'nyu' or args.dataset == 'nyud':
                        eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)
            
            # Calculate metrics #
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:measures_size - 1] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[measures_size - 1] += 1

            # For uncertainty #
            if args.predict_unc:
                unc_error = compute_error_uncertainty(gt_depth[valid_mask], pred_depth[valid_mask], unc_decoder[valid_mask])
                eval_measures_unc[0] += torch.tensor(unc_error).cuda(device=gpu)
            else:
                unc_error = None

        # Gather measures from other nodes/gpus #
        if args.multiprocessing_distributed:
            # group = dist.new_group([i for i in range(ngpus)])
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)
            if args.predict_unc:
                dist.all_reduce(tensor=eval_measures_unc, op=dist.ReduceOp.SUM, group=group)

        # Devide by sum for multiprocessing #
        if not args.multiprocessing_distributed or gpu == 0:
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[measures_size - 1].item()
            eval_measures_cpu /= cnt

            print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
            print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                         'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                         'd3'))
            for i in range(measures_size - 2):
                print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')

            print('{:7.4f}'.format(eval_measures_cpu[measures_size - 2]))

            if args.predict_unc:
                print("Eval uncertainty decoder:", unc_error)
            
            return eval_measures_cpu, unc_error

        return None
