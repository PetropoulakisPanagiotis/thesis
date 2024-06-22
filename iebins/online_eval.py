from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist

from networks.NewCRFDepth import NewCRFDepth
from utils import compute_errors, compute_error_uncertainty, sigma_metric_from_canonical_and_scale


def online_eval(args, model, dataloader_eval, gpu, epoch, ngpus, group, original_d3vo=False):
    with torch.no_grad():
        # Save results #
        measures_size = 10
        eval_measures = torch.zeros(measures_size).cuda(device=gpu)
        eval_d3vo = torch.zeros(1).cuda(device=gpu)

        for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))

            if args.segmentation:
                segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda(
                    args.gpu, non_blocking=True))

            if args.instances:
                instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda(
                    args.gpu, non_blocking=True))
                boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda(args.gpu, non_blocking=True))
                labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda(
                    args.gpu, non_blocking=True))

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

            # Mask predictions #
            if args.instances:
                #instances[:, 6:, :, :] = 0
                pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
            elif args.segmentation:
                pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
            else:
                pred_depth = pred_depths_r_list[-1]

            pred_depth = pred_depth.cpu().numpy().squeeze()

            if args.predict_unc:

                """
                if args.d3vo_c:
                    if not args.instances:
                        sigma_metric = sigma_metric_from_canonical_and_scale(result["pred_depths_rc_list"][-1],
                                                                             result["unc_d3vo_c"],
                                                                             result["pred_scale_list"][-1],
                                                                             result["unc_d3vo"], args)
                    else:
                        sigma_metric = sigma_metric_from_canonical_and_scale(
                            result["pred_depths_instances_rc_list"][-1], result["unc_d3vo_c"],
                            result["pred_scale_instances_list"][-1], result["unc_d3vo"], args)
                else:
                    # uncertainty of canonical is std --> convert to variance
                    sigma_metric = sigma_metric_from_canonical_and_scale(result["pred_depths_rc_list"][-1],
                                                                         result["uncertainty_maps_list"][-1]**2,
                                                                         result["pred_scale_list"][-1],
                                                                         result["unc_d3vo"], args)

                if args.instances:
                    sigma_metric = torch.sum((sigma_metric * instances), dim=1).squeeze(0).cpu().numpy()
                elif args.segmentation:
                    sigma_metric = torch.sum((sigma_metric * segmentation_map), dim=1).squeeze(0).cpu().numpy()
                else:
                    sigma_metric = sigma_metric.squeeze(0).squeeze(0).cpu().numpy()
                """
                if args.instances:
                    pass
                elif args.segmentation:
                    pass
                else:
                    if args.unc_head:
                        sigma_metric = sigma_metric_from_canonical_and_scale(result["pred_depths_rc_list"][-1],
                                                                             result["unc_c"][-1],
                                                                             result["pred_scale_list"][-1],
                                                                             result["unc_s"], args)
                    else:
                        # uncertainty of canonical is std --> convert to variance
                        sigma_metric = sigma_metric_from_canonical_and_scale(result["pred_depths_rc_list"][-1],
                                                                             result["uncertainty_maps_list"][-1]**2,
                                                                             result["pred_scale_list"][-1],
                                                                             result["uncertainty_maps_scale_list"]**2, args)
                    sigma_metric = sigma_metric.squeeze(0).squeeze(0).cpu().numpy()
            # Mask gt_depth #
            if args.instances:
                mask = torch.sum(instances, dim=1).squeeze(0).to(torch.bool).cpu()
                gt_depth = (gt_depth * mask).cpu().numpy().squeeze()
            else:
                gt_depth = gt_depth.cpu().numpy().squeeze()

            # Filter predicted depth #
            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            # Final mask to evaluate #
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            if args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1
                elif args.dataset == 'scannet':
                    eval_mask[:, :] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

            # Calculate metrics #
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:measures_size - 1] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[measures_size - 1] += 1

            if args.predict_unc:
                eval_d3vo_numpy = compute_error_uncertainty(gt_depth[valid_mask], pred_depth[valid_mask],
                                                            sigma_metric[valid_mask], original_d3vo)
                eval_d3vo += torch.tensor(eval_d3vo_numpy).cuda(device=gpu)

        # Gather measures from other nodes/gpus #
        if args.multiprocessing_distributed:
            # group = dist.new_group([i for i in range(ngpus)])
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)
            if args.predict_unc:
                dist.all_reduce(tensor=eval_d3vo, op=dist.ReduceOp.SUM, group=group)

        # Devide by sum for multiprocessing #
        if not args.multiprocessing_distributed or gpu == 0:
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[measures_size - 1].item()
            eval_measures_cpu /= cnt

            print('Computing errors for {} eval samples'.format(int(cnt)))
            print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
                'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
            for i in range(measures_size - 2):
                print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')

            print('{:7.4f}'.format(eval_measures_cpu[measures_size - 2]))

            if args.predict_unc:
                eval_d3vo = eval_d3vo.cpu().numpy()[0]
                eval_d3vo /= cnt
                return eval_measures_cpu, eval_d3vo
            else:
                return eval_measures_cpu, None

        return None, None
