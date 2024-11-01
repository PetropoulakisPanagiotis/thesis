import random

import torch
import cv2

from utils import inv_normalize, find_indexes_valid_instances, colormap


def tb_visualization(writer, global_step, args, current_loss_depth, current_lr, var_sum, var_cnt, num_images, \
                     depth_gt, image, max_tree_depth, pred_depths_r_list, pred_depths_rc_list, \
                     num_semantic_classes, instances, segmentation_map, \
                     labels, pred_depths_c_list, uncertainty_maps_list, \
                     pred_depths_u_list, expensive_viz=True):

    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e-3, depth_gt)

    if args.loss_type == 0 and current_loss_depth is not None:
        writer.add_scalar('silog_loss', current_loss_depth, global_step)
    elif current_loss_depth is not None:
        writer.add_scalar('l1_loss', current_loss_depth, global_step)

    if current_lr is not None:
        writer.add_scalar('learning_rate', current_lr, global_step)

    if var_sum is not None:
        writer.add_scalar('var_average', var_sum / var_cnt, global_step)

    if args.instances:
        for i in range(num_images):

            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i),
                             colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)

            # Metric #
            writer.add_image('depth_metric_est{}/image/{}'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_r_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Canonical #
            writer.add_image('depth_canonical_est{}/image/{}/'.format(0, i),
                              colormap(torch.log10(torch.sum(pred_depths_rc_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                              name='magma'), global_step)

            # Bins labels #
            writer.add_image('depth_labels_est{}/image/{}/'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_c_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            
            # Bins uncertainty #
            writer.add_image(
                'uncer_bins_est{}/image/{}/'.format(0, i),
                colormap(
                    torch.sum(uncertainty_maps_list[-1][i, :, :, :] * instances[i, :, :, :],
                              dim=0).clamp(min=1e-3).unsqueeze(0).data), global_step)
            
            # Expensive visualization #
            if expensive_viz:
                max_viz_instances = 5 # Visualize some instances predictions
                valid_indexes = find_indexes_valid_instances(labels[i])
                try:
                    if len(valid_indexes) < max_viz_instances:
                        max_viz_instances = len(valid_indexes)

                    list_valid_instances = list(valid_indexes.detach().cpu().numpy())
                    selected_instances = random.sample(list_valid_instances, max_viz_instances)

                    for idx, j in enumerate(selected_instances):
                        # Metric #
                        writer.add_image('depth_metric_est{}/image/{}/instance{}'.format(0, i, idx),
                                         colormap(torch.log10((pred_depths_r_list[-1][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data),\
                                         name='magma'), global_step)
                        # Canonical #
                        writer.add_image('depth_canonical_est{}/image/{}/instance{}'.format(0, i, idx),
                                          colormap(torch.log10((pred_depths_rc_list[-1][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                          name='magma'), global_step)
                except:
                    pass

    elif args.segmentation:
        for i in range(num_images):

            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i),
                             colormap(torch.log10(depth_gt[i, :, :, :].clamp(min=1e-3).data), name='magma'),
                             global_step)

            # Metric #
            writer.add_image('depth_metric_est{}/image/{}'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_r_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Canonical #
            writer.add_image('depth_canonical_est{}/image/{}/'.format(0, i),
                              colormap(torch.log10(torch.sum(pred_depths_rc_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                              name='magma'), global_step)

            # Bins labels #
            writer.add_image('depth_labels_est{}/image/{}/'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_c_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Bins uncertainty #
            writer.add_image(
                'uncer_bins_est{}/image/{}/'.format(0, i),
                colormap(
                    torch.sum(uncertainty_maps_list[-1][i, :, :, :] * segmentation_map[i, :, :, :],
                              dim=0).clamp(min=1e-3).unsqueeze(0).data), global_step)

            # Expensive visualization #
            if expensive_viz:
                for j in range(num_semantic_classes):
                    # Metric #
                    writer.add_image('depth_metric_est{}/image/{}/class{}'.format(0, i, j),
                                     colormap(torch.log10((pred_depths_r_list[-1][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                     name='magma'), global_step)
                    # Canonical #
                    writer.add_image('depth_canonical_est{}/image/{}/class{}'.format(0, i, j),
                                      colormap(torch.log10((pred_depths_rc_list[-1][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                      name='magma'), global_step)

    # Global scale or IEBINS #  
    else:
        for i in range(num_images):

            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i),
                             colormap(torch.log10(depth_gt[i, :, :, :].clamp(min=1e-3).data), name='magma'),
                             global_step)

            # Metric
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_r_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                 name='magma'), global_step)

            # Canonical #
            if args.update_block == 1:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_rc_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                     name='magma'), global_step)

            # Bins: Uncertainty either for IEBINS metric depth or canonical depth uncertainty #
            for ii in range(max_tree_depth):
                writer.add_image('depth_labels_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_c_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                 name='magma'), global_step)

                writer.add_image('uncer_bins_est{}/image/{}'.format(ii, i),
                                 colormap(uncertainty_maps_list[ii][i, :, :, :].clamp(min=1e-3).data), global_step)


def tb_visualization_unc(writer, current_loss_unc, sigma_metric, global_step, args, current_lr, var_sum, var_cnt, num_images, \
                     depth_gt, image, max_tree_depth, pred_depths_r_list, pred_depths_rc_list, \
                     num_semantic_classes, instances, segmentation_map, \
                     labels, pred_depths_c_list, uncertainty_maps_list, \
                     pred_depths_u_list):

    if current_loss_unc is not None:
        writer.add_scalar('unc_loss', current_loss_unc, global_step)

    if current_lr is not None:
        writer.add_scalar('learning_rate', current_lr, global_step)

    if var_sum is not None:
        writer.add_scalar('var_average', var_sum / var_cnt, global_step)

    for i in range(num_images):
        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
        writer.add_image('unc_unc_est/image/{}'.format(i),
                         colormap(sigma_metric[i, :, :, :].clamp(min=1e-3).data), global_step)
    
    if args.instances:
        for i in range(num_images):

            # Metric #
            writer.add_image('depth_metric_est{}/image/{}'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_r_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Canonical #
            writer.add_image('depth_canonical_est{}/image/{}/'.format(0, i),
                              colormap(torch.log10(torch.sum(pred_depths_rc_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                              name='magma'), global_step)

            # Bins labels #
            writer.add_image('depth_labels_est{}/image/{}/'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_c_list[-1][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            
            # Bins uncertainty #
            writer.add_image(
                'uncer_bins_est{}/image/{}/'.format(0, i),
                colormap(
                    torch.sum(uncertainty_maps_list[-1][i, :, :, :] * instances[i, :, :, :],
                              dim=0).clamp(min=1e-3).unsqueeze(0).data), global_step)
    elif args.segmentation:
        for i in range(num_images):
            # Metric #
            writer.add_image('depth_metric_est{}/image/{}'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_r_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Canonical #
            writer.add_image('depth_canonical_est{}/image/{}/'.format(0, i),
                              colormap(torch.log10(torch.sum(pred_depths_rc_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                              name='magma'), global_step)

            # Bins labels #
            writer.add_image('depth_labels_est{}/image/{}/'.format(0, i),
                             colormap(torch.log10(torch.sum(pred_depths_c_list[-1][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                             name='magma'), global_step)

            # Bins uncertainty #
            writer.add_image(
                'uncer_bins_est{}/image/{}/'.format(0, i),
                colormap(
                    torch.sum(uncertainty_maps_list[-1][i, :, :, :] * segmentation_map[i, :, :, :],
                              dim=0).clamp(min=1e-3).unsqueeze(0).data), global_step)

    # Global scale or IEBINS #  
    else:
        for i in range(num_images):
            # Metric
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_r_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                 name='magma'), global_step)

            # Canonical #
            if args.update_block == 1:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_rc_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                     name='magma'), global_step)

            # Bins: Uncertainty either for IEBINS metric depth or canonical depth uncertainty #
            for ii in range(max_tree_depth):
                writer.add_image('depth_labels_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_c_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                 name='magma'), global_step)

                writer.add_image('uncer_bins_est{}/image/{}'.format(ii, i),
                                 colormap(uncertainty_maps_list[ii][i, :, :, :].clamp(min=1e-3).data), global_step)


def debug_visualize_gt_instances(instances, mask, depth_gt):
    instances[:, 2:, :, :] = 0 # Visualize first instance for first batch

    instances_gt_mask = torch.sum(instances, dim=1).unsqueeze(1).to(torch.bool)

    mask = mask * instances_gt_mask
    depth_gt = depth_gt * instances_gt_mask

    cv2.imshow("GT depth for instance (0 of batch 0)", depth_gt[0, 0, :, :].cpu().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_result(result, gt_depth):
    if False:
        print("Metric depth:")
        print(torch.max(result['pred_depths_r_list'][-1][0, :, :, :]))
        print(torch.min(result['pred_depths_r_list'][-1][0, :, :, :]))

    if False:
        print("Canonical depth:")
        print(torch.max(result['pred_depths_rc_list'][-1][:, :, :, :]))
        print(torch.min(result['pred_depths_rc_list'][-1][:, :, :, :]))
        print(torch.mean(result['pred_depths_rc_list'][-1][:, :, :, :]))
        print(torch.std(result['pred_depths_rc_list'][-1][:, :, :, :]))

    if False:
        print("Bins uncertainty:")
        print(torch.max(result['uncertainty_maps_list'][-1][0, 0, :, :]))
        print(torch.min(result['uncertainty_maps_list'][-1][0, 0, :, :]))
        print(torch.mean(result['uncertainty_maps_list'][-1][0, 0, :, :]))
        print(torch.std(result['uncertainty_maps_list'][-1][0, 0, :, :]))

    if False:
        print("Scale:")
        print(torch.max(result['pred_scale_list'][-1][:, :]))
        print(torch.min(result['pred_scale_list'][-1][:, :]))
        print(torch.mean(result['pred_scale_list'][-1][:, :]))
        print(torch.std(result['pred_scale_list'][-1][:, :]))

    if False:
        print("Instances scale:")
        print(torch.max(result['pred_scale_instances_list'][-1][:, :]))
        print(torch.min(result['pred_scale_instances_list'][-1][:, :]))
        print(torch.mean(result['pred_scale_instances_list'][-1][:, :]))
        print(torch.std(result['pred_scale_instances_list'][-1][:, :]))

        print("Instances canonical:")
        print(torch.max(result['pred_depths_instances_rc_list'][-1][::, :, :]))
        print(torch.min(result['pred_depths_instances_rc_list'][-1][:, :, :, :]))
        print(torch.mean(result['pred_depths_instances_rc_list'][-1][:, :, :, :]))
        print(torch.std(result['pred_depths_instances_rc_list'][-1][:, :, :, :]))
