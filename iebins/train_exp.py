import os, sys, time
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import random
import cv2

from tensorboardX import SummaryWriter

from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from networks.NewCRFDepth_exp import NewCRFDepth
from networks.depth_update import *
from datetime import datetime
from utils import post_process_depth, flip_lr, silog_loss, l1_loss, l1_d3vo_loss, compute_errors, compute_errors_uncertainty, \
                    eval_metrics, entropy_loss, colormap, \
                    block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, train_parser, find_indexes_valid_instances

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = train_parser.parse_args([arg_filename_with_prefix])
else:
    args = train_parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'nyud':
    from dataloaders.dataloader_exp import NewDataLoader

def online_eval(model, update_block, dataloader_eval, gpu, epoch, ngpus, group, post_process=False):
    measures_size = 10
    eval_measures = torch.zeros(measures_size).cuda(device=gpu)
    eval_measures_unc = torch.zeros(1).cuda(device=gpu)

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            if args.dataset == 'nyu' and args.segmentation:
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
            if args.update_block >= 9 and args.update_block < 18 or args.update_block >= 20:
                if args.instances:
                    result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
                else:
                    result = model(image, masks=segmentation_map)
            else:
                result = model(image)

            pred_depths_r_list = result["pred_depths_r_list"]
            if args.instances: ### fix aggregation after
                pred_depths_r_list = result["pred_depths_instances_r_list"] 
            
            # uncertainty #
            if args.predict_unc == True:
                unc = result["unc"].cpu().numpy().squeeze()
            elif args.predict_unc_d3vo == True:
                unc = result["unc_d3vo"].cpu().numpy().squeeze()

            #  Predict depth of flipped image too and fuse - True
            if post_process and False:
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
            else:
                if args.segmentation:
                    if args.instances:
                        #instances[:, 6:, :, :] = 0
                        pred_depth = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(0)
                    else:
                        pred_depth = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(0)
                else:
                    pred_depth = pred_depths_r_list[-1]

                pred_depth = pred_depth.cpu().numpy().squeeze()

            if args.instances:
                mask = torch.sum(instances, dim=1).unsqueeze(-1).to(torch.bool).cpu()
                gt_depth = (gt_depth * mask).cpu().numpy().squeeze()
            else:
                gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop: # Not in NYU
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        # Filter depth #
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop: # NYU
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu' or args.dataset == 'nyud':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
        
        # Calculate metrics #
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        if args.predict_unc:
            unc_error = compute_errors_uncertainty(gt_depth[valid_mask], pred_depth[valid_mask], unc[valid_mask], 0)
            eval_measures_unc[0] += torch.tensor(unc_error).cuda(device=gpu)
        elif args.predict_unc_d3vo:
            unc_error = compute_errors_uncertainty(gt_depth[valid_mask], pred_depth[valid_mask], unc[valid_mask], 1)
            eval_measures_unc[0] += torch.tensor(unc_error).cuda(device=gpu)

        eval_measures[:measures_size - 1] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[measures_size - 1] += 1

    if args.multiprocessing_distributed:
        # group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)
        if args.predict_unc or args.predict_unc_d3vo:
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

        if args.predict_unc or args.predict_unc_d3vo:
            print("Eval uncer:", unc_error)
            return eval_measures_cpu, unc_error
        else:
            return eval_measures_cpu

    return None

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Dataloaders #
    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')
    num_semantic_classes = dataloader.num_semantic_classes
    num_semantic_classes = 14
    num_instances = 63
 
    # Model #
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type, 
                        train_decoder=args.train_decoder, pretrained=args.pretrain, predict_unc=args.predict_unc, 
                        predict_unc_d3vo=args.predict_unc_d3vo, num_semantic_classes=num_semantic_classes, num_instances=num_instances, var=args.var, padding_instances=args.padding_instances)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    # Nodes #
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    # Single #
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    best_unc = np.inf

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate)

    # Load checkpoint #
    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            
            # Fix weights #
            if args.update_block != 0:
                # Canonical #
                weights_to_remove_1 = "update"
                weights_to_remove_2 = "project"
                keys_to_remove = [key for key in checkpoint['model'].keys() if weights_to_remove_1 in key]
                keys_to_remove.extend([key for key in checkpoint['model'].keys() if weights_to_remove_2 in key]) 

                for key_to_remove in keys_to_remove:
                    checkpoint['model'].pop(key_to_remove)

                model.load_state_dict(checkpoint['model'], strict=False)
                #optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True



    # ===== Evaluation before training ======
    # model.eval()
    # with torch.no_grad():
    #     eval_measures = online_eval(model, args.update_block, dataloader_eval, gpu, ngpus_per_node, post_process=True)

    # Logging #
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        print(args.log_directory + '/' + args.model_name + '/summaries') 
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        
        # Metrics #
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
        
        # Log hparams #
        hparams = {
            "epochs": args.num_epochs,
            "loss_type": args.loss_type,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "update_block": args.update_block,
            "max_tree_depth": args.max_tree_depth,
            "bin_num": args.bin_num,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "train_decoder": args.train_decoder,
            "predict_unc": args.predict_unc,
            "num_semantic_classes": num_semantic_classes,
            "segmentation": args.segmentation,
            "instances": args.instances,
            "var": args.var,
            "padding_instances": args.padding_instances,
        }

        writer.add_hparams(hparam_dict=hparams, metric_dict={})
    
    # Losses #
    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    l1_criterion = l1_loss()
    l1_d3vo_criterion = l1_d3vo_loss()
    
    start_time = time.time()
    duration = 0

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate
    
    # Print stats #
    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)
    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    ii = 0
    group = dist.new_group([i for i in range(ngpus_per_node)])

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            before_op_time = time.time()
            current_loss_d = 0
            current_loss_u = 0

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            b, _, h, w = depth_gt.shape

            if args.dataset == 'nyu' and args.segmentation:
                segmentation_map = torch.autograd.Variable(sample_batched['segmentation_map'].cuda(args.gpu, non_blocking=True))
                if args.instances:
                    instances = torch.autograd.Variable(sample_batched['instances_masks'].cuda(args.gpu, non_blocking=True))
                    boxes = torch.autograd.Variable(sample_batched['instances_bbox'].cuda(args.gpu, non_blocking=True))
                    labels = torch.autograd.Variable(sample_batched['instances_labels'].cuda(args.gpu, non_blocking=True))

            num_images = image.shape[0]
            
            # Predict #            
            if args.update_block >= 9 and args.update_block < 18 or args.update_block >= 20:
                if args.instances:
                    result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
                else:
                    result = model(image, masks=segmentation_map)
            else:
                result = model(image, epoch, step)
            
            # Unpack #            
            pred_depths_r_list = result["pred_depths_r_list"]
            max_tree_depth = len(pred_depths_r_list)
            if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and args.update_block != 11 and args.update_block != 12 and args.update_block != 13 and args.update_block != 14 and args.update_block != 15 and \
               args.update_block != 16 and args.update_block != 17 and args.update_block != 20 and args.update_block != 21 and args.update_block != 22 and args.update_block != 23 and args.update_block != 25:            
                pred_depths_c_list = result["pred_depths_c_list"]
                uncertainty_maps_list = result["uncertainty_maps_list"]
            if args.instances:
                pred_depths_instances_rc_list = result["pred_depths_instances_rc_list"]
                pred_depths_instances_r_list = result["pred_depths_instances_r_list"]

                pred_scale_instances_list = result["pred_scale_instances_list"]
                pred_shift_instances_list = result["pred_shift_instances_list"]
                
                #non_zero_idx = (labels != -1).nonzero(as_tuple=False)
                #print(pred_scale_instances_list[0][non_zero_idx[:,0], non_zero_idx[:,1]].detach().cpu().numpy())
                #print(pred_shift_instances_list[0][non_zero_idx[:,0], non_zero_idx[:,1]].detach().cpu().numpy())
                #print(pred_scale_instances_list[0].detach().cpu().numpy())
                #print(pred_shift_instances_list[0].detach().cpu().numpy())
            
            # Canonical #
            if args.update_block != 0:
                pred_depths_rc_list = result["pred_depths_rc_list"]
            
            # uncertainty (GRU)
            if args.update_block == 2:
                pred_depths_u_list = result["pred_depths_u_list"]

            if args.predict_unc == True:
                unc = result["unc"]
            elif args.predict_unc_d3vo == True:
                unc_d3vo = result["unc_d3vo"]
            
            # gt_depth masking #
            if args.dataset == 'nyu' or args.dataset == 'nyud':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            # Loss #
            for curr_tree_depth in range(max_tree_depth):
                if args.segmentation:
                    if args.instances:
                        #instances[:, 6:, :, :] = 0
                        pred_d = torch.sum((pred_depths_instances_r_list[curr_tree_depth] * instances), dim=1).unsqueeze(1)
                        
                        instances_gt_mask = torch.sum(instances, dim=1).unsqueeze(1).to(torch.bool)
                        mask = mask * instances_gt_mask 
                        #image_masked_with_instances = torch.sum(instances[0, :, :, :], dim=0)
                        #depth_gt = depth_gt * instances_gt_mask
                        #cv2.imshow("instances_mapped_image", depth_gt[0,0,:,:].cpu().numpy())
                        #tmp = mask[0].permute(1,2,0).squeeze()
                        #tmp = (tmp.cpu().detach().numpy() * 255).astype('uint8')
                        #print(tmp.shape)
                        #print(instances[0, :, :, :].shape)
                        #cv2.imshow("instances_mapped_ittmage", tmp)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                    else:
                        pred_d = torch.sum((pred_depths_r_list[curr_tree_depth] * segmentation_map), dim=1).unsqueeze(1)
                else:
                    pred_d = pred_depths_r_list[curr_tree_depth]
                
                if args.loss_type == 0:
                    current_loss_d += silog_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))
                elif args.loss_type == 1:
                    current_loss_d += l1_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))
                elif args.loss_type == 2: # l1 + d3vo 
                    current_loss_d += l1_d3vo_criterion.forward(pred_d, depth_gt, unc_d3vo, mask.to(torch.bool))
           
                # uncertainty loss #
                if args.update_block == 2:
                    u_gt = torch.exp(-5 * torch.abs(depth_gt - pred_depths_r_list[curr_tree_depth].detach()) / (depth_gt + pred_depths_r_list[curr_tree_depth].detach() + 1e-7))
                    current_loss_u += torch.abs(pred_depths_u_list[curr_tree_depth][mask.to(torch.bool)] - u_gt[mask.to(torch.bool)]).mean() 
                elif args.predict_unc:           
                    u_gt = torch.exp(-5 * torch.abs(depth_gt - pred_depths_r_list[0].detach()) / (depth_gt + pred_depths_r_list[0].detach() + 1e-7))
                    current_loss_u = torch.abs(unc[mask.to(torch.bool)] - u_gt[mask.to(torch.bool)]).mean() 
                else:
                    pass

            if args.update_block == 2 or args.predict_unc:
                loss = current_loss_d + (args.uncertainty_weight * current_loss_u)
            else:
                loss = current_loss_d

            # Optimize #
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr
            optimizer.step()

            # LR + Loss #
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if args.update_block == 2 or args.predict_unc:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}, depth_loss: {:.12f}, u_loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss, current_loss_d, current_loss_u))
                else:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))

            duration += time.time() - before_op_time

            # Logging #
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                
                examples_per_sec = args.batch_size / duration * args.log_freq
                
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    if args.loss_type == 0:
                        writer.add_scalar('silog_loss', current_loss_d, global_step)
                    else:
                        writer.add_scalar('l1_loss', current_loss_d, global_step)

                    if args.update_block == 2 or args.predict_unc:
                        writer.add_scalar('u_loss_gru', current_loss_u, global_step)

                    # writer.add_scalar('var_loss', var_loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var_average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e-3, depth_gt)
           
                    if args.instances:
                        for i in range(num_images):
                            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                            writer.add_image('depth_gt/image/{}'.format(i), colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)
                            for ii in range(max_tree_depth):
                                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), 
                                                 colormap(torch.log10(torch.sum(pred_depths_instances_r_list[ii][i, :, :, :] * instances[i, :, :, :], dim=0).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)

                            if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and not (args.update_block >= 11 and args.update_block <= 17) and args.update_block != 20 and args.update_block != 21 and args.update_block != 22 and args.update_block != 23 and args.update_block != 24 and args.update_block != 25:
                                for ii in range(max_tree_depth):
                                    writer.add_image('depth_labels_est{}/image/{}/'.format(ii, i), 
                                                     colormap(torch.log10(torch.sum(pred_depths_c_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), name='magma'), global_step)
                                    writer.add_image('uncer_bins_est{}/image/{}/'.format(ii, i), 
                                                     colormap(torch.sum(uncertainty_maps_list[ii][i, :, :, :] * segmentation_map[i, :, :, :], dim=0).unsqueeze(0).data), global_step)

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
                            
                            if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and args.update_block != 11 and args.update_block != 12 and args.update_block != 13 and args.update_block != 14 and args.update_block != 15 and args.update_block != 16 and args.update_block != 17:            
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
                                    for ii in range(max_tree_depth):
                                        writer.add_image('depth_labels_est{}/image/{}/class{}'.format(ii, i, j), 
                                                         colormap(torch.log10((pred_depths_c_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), name='magma'), global_step)
                                    if args.update_block != 0:
                                        for ii in range(max_tree_depth):
                                            writer.add_image('depth_canonical_est{}/image/{}/class{}'.format(ii, i, j), 
                                                              colormap(torch.log10((pred_depths_rc_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-5).unsqueeze(0).data), name='magma'), global_step)
                                    for ii in range(max_tree_depth):
                                        writer.add_image('uncer_bins_est{}/image/{}/class{}'.format(ii, i, j), 
                                                         colormap((uncertainty_maps_list[ii][i, j, :, :] * segmentation_map[i, j, :, :]).clamp(min=1e-3).unsqueeze(0).data), global_step)
                    else:
                        for i in range(num_images):
                            writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                            # Depth #
                            writer.add_image('depth_gt/image/{}'.format(i), colormap(torch.log10(depth_gt[i, :, :, :].data), name='magma'), global_step)
                            for ii in range(max_tree_depth):
                                writer.add_image('depth_metric_est{}/image/{}'.format(ii, i), colormap(torch.log10(pred_depths_r_list[ii][i, :, :, :].data), name='magma'), global_step)
                            if args.update_block != 7 and args.update_block != 8 and args.update_block != 10 and args.update_block != 11 and args.update_block != 12 and args.update_block != 13 and args.update_block != 14 and args.update_block != 15 and args.update_block != 16 and args.update_block != 17:            
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
                            elif args.predict_unc_d3vo:
                                writer.add_image('uncer_depth_d3vo_est{}/image/{}'.format(ii, i), colormap(unc_d3vo[i, :, :, :].data, name='viridis'), global_step)


            # Evaluate #
            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    if args.predict_unc or args.predict_unc_d3vo:
                        eval_measures, unc_error = online_eval(model, args.update_block, dataloader_eval, gpu, epoch, ngpus_per_node, group, post_process=True)
                    else:
                        eval_measures = online_eval(model, args.update_block, dataloader_eval, gpu, epoch, ngpus_per_node, group, post_process=True)

                if eval_measures is not None:
                    if args.predict_unc or args.predict_unc_d3vo:
                        if best_unc > unc_error:
                            best_unc = unc_error

                    exp_name = args.exp_name
                    log_txt = os.path.join(args.log_directory + '/' + args.model_name, exp_name+'_logs.txt')
                    with open(log_txt, 'a') as txtfile:
                        txtfile.write(">>>>>>>>>>>>>>>>>>>>>>>>>Step:%d>>>>>>>>>>>>>>>>>>>>>>>>>\n"%(int(global_step)))
                        txtfile.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format('silog', 
                                        'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2','d3'))
                        txtfile.write("depth estimation\n")
                        line = ''
                        for i in range(9):
                            line +='{:7.4f}, '.format(eval_measures[i])
                        txtfile.write(line+'\n')

                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                    print("Best evals at global step: " + str(global_step))
                    print(best_eval_measures_higher_better)
                    print(best_eval_measures_lower_better)
                    
                    if args.predict_unc or args.predict_unc_d3vo:
                        print(best_unc) 
                
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()

    print("Training ended:\n")
    print(best_eval_measures_higher_better)
    print(best_eval_measures_lower_better)
    if args.predict_unc or args.predict_unc_d3vo:
        print(best_unc)

def main():
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    exp_name = args.exp_name  
    args.log_directory = os.path.join(args.log_directory,exp_name)  
    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)
    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
        command = 'cp iebins/train.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp iebins/networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp iebins/dataloaders/*.py ' + dataloaders_savepath
        os.system(command)
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
