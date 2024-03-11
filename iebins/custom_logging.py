import torch
import random
import cv2
from utils_clean import inv_normalize, find_indexes_valid_instances, colormap


def tb_visualization(writer, global_step, args, num_images, depth_gt, image, max_tree_depth, pred_depths_r_list, pred_depths_rc_list, pred_depths_instances_r_list, pred_depths_instances_rc_list, num_semantic_classes, instances, segmentation_map, labels, pred_depths_c_list, uncertainty_maps_list, pred_depths_u_list, unc):
    if args.instances:
        for i in range(num_images):
            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i), colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), 
                                 colormap(torch.log10(torch.sum(pred_depths_instances_r_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)

            if args.update_block != 8 and not (args.update_block >= 12 and args.update_block <= 15) and args.update_block != 20 and args.update_block != 21 \
               and args.update_block != 22 and args.update_block != 23 and args.update_block != 24 and args.update_block != 25 and args.update_block != 26:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i), 
                                     colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)
                    writer.add_image('uncer_bins_est{}/image/{}/'.format(ii, i), 
                                     colormap(torch.sum(uncertainty_maps_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).unsqueeze(0).data), global_step)

            if args.update_block != 0:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i), 
                                      colormap(torch.log10(torch.sum(pred_depths_instances_rc_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)
            if True: # expensive
                max_vizualization = 5
                valid_indexes = find_indexes_valid_instances(labels[i])

                if(len(valid_indexes) < max_vizualization):
                    max_vizualization = len(valid_indexes)
                
                picked_items = random.sample(list(valid_indexes.detach().cpu().numpy()), max_vizualization)
                for idx, j in enumerate(picked_items):
                    # Depth #
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}/instance{}'.format(ii, i, idx), 
                                         colormap(torch.log10((pred_depths_instances_r_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)
                    if args.update_block != 0:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_canonical_est{}/image/{}/instance{}'.format(ii, i, idx), 
                                              colormap(torch.log10((pred_depths_instances_rc_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-5).unsqueeze(0).data), name='magma'), global_step)
    elif args.segmentation:
        for i in range(num_images):
            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            writer.add_image('depth_gt/image/{}'.format(i), colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), 
                                 colormap(torch.log10(torch.sum(pred_depths_r_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), global_step)
            if args.predict_unc:
                writer.add_image('uncer_depth_est{}/image/{}'.format(ii, i), colormap(unc[i, :, :, :].data, name='viridis'), global_step)

            else:
                if args.update_block != 8 and args.update_block != 12 \
                   and args.update_block != 13 and args.update_block != 15:            
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i), 
                                         colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), global_step)
                        writer.add_image('uncer_bins_est{}/image/{}/'.format(ii, i), 
                                         colormap(torch.sum(uncertainty_maps_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), global_step)

            if args.update_block != 0:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i), 
                                      colormap(torch.log10(torch.sum(pred_depths_rc_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), global_step)
            if True: # expensive        
                for j in range(num_semantic_classes):
                    # Depth #
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}/class{}'.format(ii, i, j), 
                                         colormap(torch.log10((pred_depths_r_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)
                    if args.update_block != 0:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_canonical_est{}/image/{}/class{}'.format(ii, i, j), 
                                              colormap(torch.log10((pred_depths_rc_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-5).unsqueeze(0).data), name='magma'), global_step)
    else:
        for i in range(num_images):
            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
            # Depth #
            writer.add_image('depth_gt/image/{}'.format(i), colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)
            for ii in range(max_tree_depth):
                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_r_list[ii][i, :, :, :].data), name='magma'), global_step)
            if args.update_block != 8 and args.update_block != 12 \
               and args.update_block != 13 and args.update_block != 15:            
                for ii in range(max_tree_depth):
                    writer.add_image('depth_labels_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_c_list[ii][i, :, :, :].data), name='magma'), global_step)
                    writer.add_image('uncer_bins_est{}/image/{}'.format(ii, i), colormap(uncertainty_maps_list[ii][i, :, :, :].data), global_step)
            
            if args.update_block != 0:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_canonical_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_rc_list[ii][i, :, :, :].data), name='magma'), global_step)
            # uncertainty #
            if args.update_block == 2:
                for ii in range(max_tree_depth):
                    writer.add_image('uncer_depth_gru_est{}/image/{}'.format(ii, i), colormap(pred_depths_u_list[ii][i, :, :, :].data), global_step)
            if args.predict_unc:
                writer.add_image('uncer_depth_est{}/image/{}'.format(ii, i), colormap(unc[i, :, :, :].data, name='viridis'), global_step)

def debug_result(result, gt_depth):
    if True: 
        if True:
            print("depth")
            print(torch.max(result['pred_depths_r_list'][-1][0, :, :, :]))
            print(torch.min(result['pred_depths_r_list'][-1][0, :, :, :]))
        if True:
            print("canonical")
            print(torch.max(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.min(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.mean(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.std(result['pred_depths_rc_list'][-1][:, :, :, :]))
        if False:
            print("uncertainty (std)")
            print(torch.max(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.min(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.mean(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.std(result['uncertainty_maps_list'][-1][0, 0, :, :]))            
        if True:
            print("scale")
            print(torch.max(result['pred_scale_list'][-1][:, :]))
            print(torch.min(result['pred_scale_list'][-1][:, :]))
            print(torch.mean(result['pred_scale_list'][-1][:, :]))
            print(torch.std(result['pred_scale_list'][-1][:, :]))            
        if True:
            print("shift")
            print(torch.max(result['pred_shift_list'][-1][:, :]))
            print(torch.min(result['pred_shift_list'][-1][:, :]))
            print(torch.mean(result['pred_shift_list'][-1][:, :]))
            print(torch.std(result['pred_shift_list'][-1][:, :]))
        if False:
            print("sample")
            print(result['pred_depths_r_list'][-1][0, 0, 120, 701])
            print(gt_depth[0, 120, 701, 0])
        if False:
            print("instances scale")
            print(torch.max(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.min(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.mean(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.std(result['pred_scale_instances_list'][-1][0, 0]))  
            print("instances shift")
            print(torch.max(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.min(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.mean(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.std(result['pred_shift_instances_list'][-1][0, 0]))  
            print("instances canonical")
            print(torch.max(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.min(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.mean(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.std(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))

def debug_visualize_masks(instances, pred_depths_instances_r_list, mask, depth_gt, ):
    instances[:, 6:, :, :] = 0
    
    pred_d = torch.sum((pred_depths_instances_r_list[0] * instances), dim=1).unsqueeze(1)
    instances_gt_mask = torch.sum(instances, dim=1).unsqueeze(1).to(torch.bool)
    
    mask = mask * instances_gt_mask 
    image_masked_with_instances = torch.sum(instances[0, :, :, :], dim=0)
    depth_gt = depth_gt * instances_gt_mask
    
    cv2.imshow("instances_mapped_image", depth_gt[0,0,:,:].cpu().numpy())
    tmp = mask[0].permute(1,2,0).squeeze()
    tmp = (tmp.cpu().detach().numpy() * 255).astype('uint8')
    
    print(tmp.shape)
    print(instances[0, :, :, :].shape)
    
    cv2.imshow("instances_mapped_ittmage", tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

