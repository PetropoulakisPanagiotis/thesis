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

from networks.NewCRFDepth import NewCRFDepth
from utils import post_process_depth, flip_lr, silog_loss, l1_loss, compute_errors, eval_metrics, \
                  entropy_loss, colormap, block_print, enable_print, normalize_result, inv_normalize, \
                  convert_arg_line_to_args, eval_parser, find_indexes_valid_instances, debug_result

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
elif sys.argv.__len__() == 3:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
    args.pick_class = torch.tensor(int(sys.argv[2]))
else:
    args = eval_parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader

def tb_visualization(writer, step, gt_depth, image, result, args, instances=None, segmentation_map=None, labels=None, num_semantic_classes=14):
    pred_depths_r_list = result["pred_depths_r_list"]
    if args.instances:
        pred_depths_instances_rc_list = result["pred_depths_instances_rc_list"]
        pred_depths_instances_r_list = result["pred_depths_instances_r_list"]
    
    if False:
        if args.update_block != 7 and args.update_block != 8:            
            pred_depths_c_list = result["pred_depths_c_list"]
            uncertainty_maps_list = result["uncertainty_maps_list"]
    if args.update_block != 0:
        pred_depths_rc_list = result["pred_depths_rc_list"]
    if args.update_block == 2:
        pred_depths_u_list = result["pred_depths_u_list"]
    if args.predict_unc == True:
        unc = result["unc"]

    max_tree_depth = len(pred_depths_r_list)
    num_log_images = image.shape[0]
    for i in range(num_log_images):
        if args.dataset == 'nyu':
            writer.add_image('gt_depth/image/{}'.format(i), colormap(torch.where(gt_depth < 1e-3, gt_depth * 0 + 1e-3, gt_depth)[i, :, :, :].data), step)
            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, step)
            if args.instances:
                for i in range(num_log_images):
                    writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, step)
                    writer.add_image('gt_depth/image/{}'.format(i), colormap(torch.log10(gt_depth[i, :, :, :].data), name='magma'), step)
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), 
                                         colormap(torch.log10(torch.sum(pred_depths_instances_r_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), step)
                    if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and not (args.update_block >= 11 and args.update_block <= 17) and args.update_block != 20:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i), 
                                             colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), step)
                            writer.add_image('uncer_bins_est{}/image/{}/'.format(ii, i), 
                                             colormap(torch.sum(uncertainty_maps_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), step)
                    if args.update_block != 0:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i), 
                                              colormap(torch.log10(torch.sum(pred_depths_instances_rc_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), step)
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
                                                 colormap(torch.log10((pred_depths_instances_r_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), name='magma'), step)
                            if args.update_block != 0:
                                for ii in range(max_tree_depth):
                                    writer.add_image('depth_canonical_est{}/image/{}/instance{}'.format(ii, i, idx), 
                                                      colormap(torch.log10((pred_depths_instances_rc_list[ii][i, j, :, :] * instances[i, j, :, :]).clamp(min=1e-5).unsqueeze(0).data), name='magma'), step)
            elif args.segmentation:
                for i in range(num_log_images):
                    writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, step)
                    writer.add_image('gt_depth/image/{}'.format(i), colormap(torch.log10(gt_depth[i, :, :, :].data), name='magma'), step)
                    for ii in range(max_tree_depth):
                        writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), 
                                         colormap(torch.log10(torch.sum(pred_depths_r_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), step)
                    if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and not (args.update_block >= 11 and args.update_block <= 17) and args.update_block != 20:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i), 
                                             colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), step)
                            writer.add_image('uncer_bins_est{}/image/{}/'.format(ii, i), 
                                             colormap(torch.sum(uncertainty_maps_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), step)
                    if args.update_block != 0:
                        for ii in range(max_tree_depth):
                            writer.add_image('depth_canonical_est{}/image/{}/'.format(ii, i), 
                                              colormap(torch.log10(torch.sum(pred_depths_rc_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), step)
                    if True: # expensive        
                        for j in range(num_semantic_classes):
                            # Depth #
                            for ii in range(max_tree_depth):
                                writer.add_image('depth_metric_est{}/image/{}/class{}'.format(ii, i, j), 
                                                 colormap(torch.log10((pred_depths_r_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), name='magma'), step)
                            if args.update_block != 0:
                                for ii in range(max_tree_depth):
                                    writer.add_image('depth_canonical_est{}/image/{}/class{}'.format(ii, i, j), 
                                                      colormap(torch.log10((pred_depths_rc_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-5).unsqueeze(0).data), name='magma'), step)

            else:
                for ii in range(max_tree_depth):
                    pred_depths_r_list[ii] = torch.sum((pred_depths_r_list[ii] * segmentation_map), dim=1).unsqueeze(1)
                    writer.add_image('depth_r_est0/image/{}'.format(i), colormap(pred_depths_r_list[ii][i, :, :, :].data), step)
        else:
            gt_depth_viz  = torch.where(gt_depth < 1e-3, gt_depth * 0 + 1e-3, gt_depth)
            gt_depth_viz = gt_depth_viz.permute(0, 3, 1, 2)
            
            writer.add_image('gt_depth/image/{}'.format(i), colormap(torch.log10(gt_depth_viz[i, :, :, :].data), name='magma'), step)
            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, step)
            for ii in range(max_tree_depth):
                writer.add_image('depth_r_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_r_list[ii][i, :, :, :].data), name='magma'), step)
            for ii in range(max_tree_depth):
                writer.add_image('depth_c_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_c_list[ii][i, :, :, :].data), name='magma'), step)

            if args.update_block != 0:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_rc_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_rc_list[ii][i, :, :, :].data), name='magma'), step)

            if args.update_block == 2:
                for ii in range(max_tree_depth):
                    writer.add_image('depth_u_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_u_list[ii][i, :, :, :].data), name='viridis'), step)

            if args.predict_unc:
                writer.add_image('unc_head_est{}/image/{}'.format(ii, i), colormap(torch.log10(unc[i, :, :, :].data), name='viridis'), step)

            if args.predict_unc_d3vo:
                writer.add_image('unc_d3vo_head_est{}/image/{}'.format(ii, i), colormap(torch.log10(unc_d3vo[i, :, :, :].data), name='viridis'), step)

        if False:
            for ii in range(max_tree_depth):
                writer.add_image('uncer_est{}/image/{}'.format(ii, i), colormap(torch.log10(uncertainty_maps_list[ii][i, :, :, :].data), name='magma'), step)

def eval_func(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()

    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_semantic_classes = 14

    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
   
    for step, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']

            if args.dataset == 'nyu' and args.segmentation:
                segmentation_map = torch.autograd.Variable(eval_sample_batched['segmentation_map'].cuda())
                instances = torch.autograd.Variable(eval_sample_batched['instances_masks'].cuda())
                boxes = torch.autograd.Variable(eval_sample_batched['instances_bbox'].cuda())
                labels = torch.autograd.Variable(eval_sample_batched['instances_labels'].cuda())
                instances_areas = torch.autograd.Variable(eval_sample_batched['instances_areas'].cuda())
            
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            # Predict #
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image)

            if False:
                debug_result(result, gt_depth)

            if False:
                # Tensorboard #
                if args.instances:
                    tb_visualization(writer, step, gt_depth, image, result, args, instances=instances, segmentation_map=segmentation_map, labels=labels)
                elif args.segmentation:
                    tb_visualization(writer, step, gt_depth, image, result, args, segmentation_map=segmentation_map)
                else:
                    tb_visualization(writer, step, gt_depth, image, result, args)

            # Unpack #            
            pred_depths_r_list = result["pred_depths_r_list"]
            if args.instances:
                pred_depths_r_list = result["pred_depths_instances_r_list"]
            
            if post_process and False:
                image_flipped = flip_lr(image)
                if args.update_block == 9:
                    result = model(image_flipped, masks = segmentation_map)
                else:
                    result = model(image_flipped)
                pred_depths_r_list_flipped = result["pred_depths_r_list"]

                pred_depths_r_list_flipped[-1] = torch.sum((pred_depths_r_list_flipped[-1] * segmentation_map), dim=1).unsqueeze(1)

                pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])
            else:
                if args.instances:
                    if args.pick_class != 0:    
                        non_class = torch.nonzero(labels[0] != args.pick_class)
                        if non_class.shape[0] == 63:
                            continue
                        
                    instances[0, non_class] = torch.zeros_like(instances[0, non_class])
                    instances_mask = torch.sum(instances, dim=1)
                    pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
                    mask = torch.sum(instances, dim=1).unsqueeze(-1).to(torch.bool).cpu()
                    gt_depth = (gt_depth * mask)
                elif args.segmentation:
                    pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
                    if True: # Fair comparison
                        if args.pick_class != 0:    
                            non_class = torch.nonzero(labels[0] != args.pick_class)
                            if non_class.shape[0] == 63:
                                continue
                            
                        instances[0, non_class] = torch.zeros_like(instances[0, non_class])
                        instances_mask = torch.sum(instances, dim=1)
                        #tmp = instances_mask.permute(1,2,0)
                        #cv2.imshow("instances_mapped_image", (tmp.cpu().numpy() * 255).astype('uint8'))
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        pred_depth = pred_depth * instances_mask
                        mask = instances_mask.unsqueeze(-1).to(torch.bool).cpu()
                        gt_depth = (gt_depth * mask)
                else:
                    pred_depth = pred_depths_r_list[-1]
                
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()     
        
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped
        
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < 3.0)
        #valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        #valid_mask = np.logical_and(gt_depth >= 5.0, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            if np.all(gt_depth[valid_mask] == 0) or len(gt_depth[valid_mask]) == 1:
                continue
        if False:
            # Calculate depth errors
            depth_errors = np.abs(gt_depth[valid_mask] - pred_depth[valid_mask])
            depth_errors = depth_errors[0:500]
            print(np.mean(depth_errors))
            print(np.max(depth_errors))
            # Plot depth errors over ground truth depth
            plt.figure(figsize=(8, 6))
            plt.scatter(gt_depth[valid_mask][0:500], depth_errors, color='blue', marker='o')
            #plt.plot(gt_depth[valid_mask][0:100], color='blue', marker='o')
            plt.title('Depth Error over Ground Truth Depth')
            plt.ylabel('Depth Error')
            plt.xlabel('Distance')
            plt.grid(True)
            plt.show()

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))

    return eval_measures_cpu


def main_worker(args):
    
    dataloader_eval = NewDataLoader(args, 'online_eval')
    num_semantic_classes = dataloader_eval.num_semantic_classes
    num_semantic_classes = 14
    num_instances = 63
    
    # depth model
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type, 
                        predict_unc=args.predict_unc, predict_unc_d3vo=args.predict_unc_d3vo, num_semantic_classes=num_semantic_classes, num_instances=num_instances, var=args.var)
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

    # evaluate #
    model.eval()
    with torch.no_grad():
        eval_measures = eval_func(model, dataloader_eval, post_process=True)

def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    exp_name = args.exp_name  

    args.log_directory = os.path.join(args.log_directory,exp_name)  
    
    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)
    print(args.log_directory)
    
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)

if __name__ == '__main__':
    main()
