import random

import torch
import cv2

from utils import inv_normalize, find_indexes_valid_instances, colormap


def tb_visualization(writer, global_step, args, current_loss_depth, current_lr, current_loss_unc_decoder, var_sum, var_cnt, num_images, \
                     depth_gt, image, max_tree_depth, pred_depths_r_list, pred_depths_rc_list, \
                     pred_depths_instances_r_list, pred_depths_instances_rc_list, num_semantic_classes, instances, segmentation_map, \
                     labels, pred_depths_c_list, uncertainty_maps_list, pred_depths_u_list, unc_decoder, expensive_viz=True):

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

            # Metric
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i),
                                 colormap(torch.log10(torch.sum(pred_depths_instances_r_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                                 name='magma'), global_step)

            # Canonical
            for ii in range(max_tree_depth):
                writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i),
                                  colormap(torch.log10(torch.sum(pred_depths_instances_rc_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                                  name='magma'), global_step)

            # Bins
            if args.update_block == 2:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i),
                                     colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), \
                                     name='magma'), global_step)

                    writer.add_image(
                        'uncer_bins_est{}/image/{}/'.format(ii, i),
                        colormap(
                            torch.sum(uncertainty_maps_list[ii][i, :, :, :] * instances[i, :, :, :],
                                      dim=0).unsqueeze(0).data), global_step)

            # Expensive visualization
            if expensive_viz:
                max_viz_instances = 5
                valid_indexes = find_indexes_valid_instances(labels[i])

                if (len(valid_indexes) < max_viz_instances):
                    max_viz_instances = len(valid_indexes)

                list_valid_instances = list(valid_indexes.detach().cpu().numpy())
                selected_instances = random.sample(list_valid_instances, max_viz_instances)

                for idx, j in enumerate(selected_instances):
                    # Metric
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}/instance{}'.format(ii, i, idx),
                                         colormap(torch.log10((pred_depths_instances_r_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data),\
                                         name='magma'), global_step)
                    # Canonical
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_canonical_est{}/image/{}/instance{}'.format(ii, i, idx),
                                          colormap(torch.log10((pred_depths_instances_rc_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                          name='magma'), global_step)

    elif args.segmentation:
        for i in range(num_images):

            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i),
                             colormap(torch.log10(depth_gt[i, :, :, :].clamp(min=1e-3).data), name='magma'),
                             global_step)

            # Metric
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i),
                                 colormap(torch.log10(torch.sum(pred_depths_r_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), \
                                 name='magma'), global_step)

            # Canonical
            for ii in range(max_tree_depth):
                writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i),
                                  colormap(torch.log10(torch.sum(pred_depths_rc_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), \
                                  name='magma'), global_step)

            # Bins
            if args.update_block == 18:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i),
                                     colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), \
                                     name='magma'), global_step)

                    writer.add_image(
                        'uncer_bins_est{}/image/{}/'.format(ii, i),
                        colormap(
                            torch.sum(uncertainty_maps_list[ii][i, :, :, :] * segmentation_map[i, :, :, :],
                                      dim=0).unsqueeze(0).data), global_step)

            # Expensive visualization
            if expensive_viz:
                for j in range(num_semantic_classes):

                    # Metric
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}/class{}'.format(ii, i, j),
                                         colormap(torch.log10((pred_depths_r_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                         name='magma'), global_step)

                    # Canonical
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_canonical_est{}/image/{}/class{}'.format(ii, i, j),
                                          colormap(torch.log10((pred_depths_rc_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), \
                                          name='magma'), global_step)

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

            # Canonical
            if args.update_block == 8 or args.update_block == 1:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_rc_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                     name='magma'), global_step)

            # Bins
            else:
                if args.update_block == 0 or args.update_block == 3 or args.update_block == 1:
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_labels_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_c_list[ii][i, :, :, :].clamp(min=1e-3).data), \
                                         name='magma'), global_step)

                        writer.add_image('uncer_bins_est{}/image/{}'.format(ii, i),
                                         colormap(uncertainty_maps_list[ii][i, :, :, :].clamp(min=1e-3).data),
                                         global_step)


def tb_visualization_d3vo(writer, global_step, args, current_loss_d3vo, current_lr, var_sum, var_cnt, num_images,
                          sigma_metric):

    if current_loss_d3vo is not None:
        writer.add_scalar('d3vo_loss', current_loss_d3vo, global_step)

    if current_lr is not None:
        writer.add_scalar('learning_rate', current_lr, global_step)

    if var_sum is not None:
        writer.add_scalar('var_average', var_sum / var_cnt, global_step)

    for i in range(num_images):
        writer.add_image('d3vo_unc_est/image/{}'.format(i),
                         colormap(sigma_metric[i, :, :, :].clamp(min=1e-3).data, name='viridis'), global_step)


def debug_visualize_gt_instances(instances, mask, depth_gt):
    instances[:, 2:, :, :] = 0

    instances_gt_mask = torch.sum(instances, dim=1).unsqueeze(1).to(torch.bool)

    mask = mask * instances_gt_mask
    depth_gt = depth_gt * instances_gt_mask

    cv2.imshow("gt instances viz", depth_gt[0, 0, :, :].cpu().numpy())
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
        print("Uncertainty decoder:")
        print(torch.max(result['unc_decoder'][0, :, :, :]))
        print(torch.min(result['unc_decoder'][0, :, :, :]))
        print(torch.mean(result['unc_decoder'][0, :, :, :]))
        print(torch.std(result['unc_decoder'][0, :, :, :]))

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
