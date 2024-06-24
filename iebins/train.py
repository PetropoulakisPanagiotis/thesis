import os, sys, time, gc
from datetime import datetime
from telnetlib import IP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

import argparse
import random
import numpy as np
import cv2

from dataloaders.dataloader import DataLoaderCustom
from networks.NewCRFDepth import NewCRFDepth
from parser_options import train_parser
from custom_logging import debug_result, debug_visualize_gt_instances, tb_visualization, tb_visualization_d3vo
from online_eval import online_eval
from utils import silog_loss, l1_loss, d3vo_loss, compute_errors, \
                    eval_metrics, block_print, enable_print, load_checkpoint_skip_update_project, load_checkpoint, \
                    set_hparams_dict, sigma_metric_from_canonical_and_scale

# Parse config file #
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = train_parser.parse_args([arg_filename_with_prefix])
else:
    args = train_parser.parse_args()


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    # Dataloaders #
    dataloader = DataLoaderCustom(args, 'train')
    dataloader_eval = DataLoaderCustom(args, 'online_eval')
    num_semantic_classes = dataloader.num_semantic_classes
    num_instances = dataloader.num_instances

    # Only in IEBINS GRU is used #
    if args.update_block != 0:
        args.max_tree_depth = 1

    # Model #
    model = NewCRFDepth(version=args.encoder, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, min_depth=args.min_depth,
                        max_depth=args.max_depth, update_block=args.update_block, loss_type=args.loss_type,
                        train_decoder=args.train_decoder, pretrained=args.pretrain,
                        num_semantic_classes=num_semantic_classes, num_instances=num_instances, \
                        padding_instances=args.padding_instances, \
                        segmentation_active=args.segmentation,  instances_active=args.instances, \
                        roi_align=args.roi_align, roi_align_size=args.roi_align_size, \
                        bins_scale=args.bins_scale, unc_head=args.unc_head, virtual_depth_variation=args.virtual_depth_variation, \
                        upsample_type=args.upsample_type, bins_type=args.bins_type, bins_type_scale=args.bins_type_scale)
    model.train()
    if args.unc_head:  # Set some layers to eval to train the uncertainty decoder
        model.set_to_eval_unc()

    # Print stats #
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    # Set device and multiprocessing #
    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        print("== Model Initialized")
        model = torch.nn.DataParallel(model)
        model.cuda()

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(10, dtype=np.int32)
    best_unc = np.inf  # Best uncertainty value

    optimizer = torch.optim.Adam([{'params': model.module.parameters()}], lr=args.learning_rate)

    # Load checkpoint and print stats #
    if args.unc_head:
        load_checkpoint(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)
    else:  # IEBINS load checkpoint - skip some layers
        if 'saved_models' in args.checkpoint_path:  # IEBINS saved models
            load_checkpoint_skip_update_project(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)
        else:
            load_checkpoint(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)
    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum / var_cnt))

    cudnn.benchmark = True

    # tb logging #
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        print("Logging dir: " + args.log_directory + '/' + args.model_name + '/summaries')
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)

        # Eval writer #
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

        # Log hparams #
        hparams = set_hparams_dict(args, num_semantic_classes)
        writer.add_hparams(hparam_dict=hparams, metric_dict={})

    # Losses #
    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    l1_criterion = l1_loss()
    d3vo_criterion = d3vo_loss(original=args.d3vo_original)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    start_time = time.time()
    duration = 0
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    group = dist.new_group([i for i in range(ngpus_per_node)])

    # Initialize vars
    pred_depths_r_list, pred_depths_rc_list, pred_depths_instances_r_list, \
        pred_depths_instances_rc_list, pred_depths_c_list, uncertainty_maps_list, \
        pred_depths_u_list = [], [], [], [], [], [], []
    segmentation_map, instances, labels, unc_decoder = None, None, None, None

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            before_op_time = time.time()
            current_loss_depth = 0
            current_loss_unc_decoder = 0
            current_loss_d3vo = 0

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            b, _, h, w = depth_gt.shape

            if args.segmentation:
                segmentation_map = torch.autograd.Variable(sample_batched['segmentation_map'].cuda(
                    args.gpu, non_blocking=True))
            if args.instances:
                instances = torch.autograd.Variable(sample_batched['instances_masks'].cuda(args.gpu, non_blocking=True))
                boxes = torch.autograd.Variable(sample_batched['instances_bbox'].cuda(args.gpu, non_blocking=True))
                labels = torch.autograd.Variable(sample_batched['instances_labels'].cuda(args.gpu, non_blocking=True))

            num_images = image.shape[0]

            # Predict #
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image, epoch, step)

            #debug_result(result, depth_gt)

            # Unpack result #
            if args.instances:
                pred_depths_instances_rc_list = result["pred_depths_instances_rc_list"]
                pred_depths_instances_r_list = result["pred_depths_instances_r_list"]
                pred_scale_instances_list = result["pred_scale_instances_list"]
            else:
                pred_depths_r_list = result["pred_depths_r_list"]

                # Canonical segmentation/single scale #
                if args.update_block != 0 and args.update_block != 4 and args.update_block != 3:
                    pred_depths_rc_list = result["pred_depths_rc_list"]
                    pred_scale_list = result["pred_scale_list"]

            # Uncertainty bins #
            pred_depths_c_list = result["pred_depths_c_list"]
            uncertainty_maps_list = result["uncertainty_maps_list"]

            if args.unc_head:  # scale unc
                unc_s = result["unc_s"][-1]
                unc_c = result["unc_c"][-1]

            # gt_depth masking #
            mask = depth_gt > 0.1

            #debug_visualize_gt_instances(instances, mask, depth_gt)

            # Train only uncertainty decoder, most of the weights are frozen #
            if args.unc_head:
                """
                if not args.instances:
                    sigma_metric = sigma_metric_from_canonical_and_scale(pred_depths_rc_list[-1], unc_c,
                                                                         pred_scale_list[-1], unc_s, args)
                elif ar:
                    sigma_metric = sigma_metric_from_canonical_and_scale(pred_depths_instances_rc_list[-1],
                                                                         unc_c, pred_scale_instances_list[-1],
                                                                         unc_s, args)
                """

                if args.instances:
                    pass
                elif args.segmentation:
                    pass
                else:
                    sigma_metric = sigma_metric_from_canonical_and_scale(pred_depths_rc_list[-1],
                                                                         unc_c,
                                                                         pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1), unc_s.unsqueeze(-1).unsqueeze(-1), args)

                if args.instances:
                    sigma_metric = torch.sum((sigma_metric * instances), dim=1).unsqueeze(1)
                    pred_d = torch.sum((pred_depths_instances_rc_list[-1] * instances), dim=1).unsqueeze(1)
                elif args.segmentation:
                    sigma_metric = torch.sum((sigma_metric * segmentation_map), dim=1).unsqueeze(1)
                    pred_d = torch.sum((pred_depths_rc_list[-1] * segmentation_map), dim=1).unsqueeze(1)
                else:
                    pred_d = pred_depths_r_list[-1]

                loss = d3vo_criterion.forward(pred_d, depth_gt, sigma_metric, mask.to(torch.bool))
                current_loss_d3vo = loss
            else:  # Depth loss #
                for curr_tree_depth in range(args.max_tree_depth):
                    if args.instances:
                        pred_d = torch.sum((pred_depths_instances_r_list[curr_tree_depth] * instances),
                                           dim=1).unsqueeze(1)
                    elif args.segmentation:
                        pred_d = torch.sum((pred_depths_r_list[curr_tree_depth] * segmentation_map), dim=1).unsqueeze(1)
                    else:
                        pred_d = pred_depths_r_list[curr_tree_depth]

                    if args.loss_type == 0:
                        current_loss_depth += silog_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))
                    elif args.loss_type == 1:
                        current_loss_depth += l1_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))

                loss = current_loss_depth

            # Optimize #
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate -
                              end_learning_rate) * (1 - global_step / num_total_steps)**0.9 + end_learning_rate
                param_group['lr'] = current_lr
            optimizer.step()

            # Print stats #
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if args.unc_head:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, d3vo_loss: {:.12f}'.format(epoch, \
                           step, steps_per_epoch, global_step, current_lr, loss))
                else:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, \
                           step, steps_per_epoch, global_step, current_lr, loss))

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

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))

                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(
                    print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(),
                                        var_sum.item() / var_cnt, time_sofar, training_time_left))

                # Tensorboard viz #
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    if args.unc_head:
                        tb_visualization_d3vo(writer, global_step, args, current_loss_d3vo, current_lr, var_sum, var_cnt, \
                                              num_images, sigma_metric)
                    else:
                        tb_visualization(writer, global_step, args, current_loss_depth, current_lr, current_loss_unc_decoder, var_sum, var_cnt,\
                                     num_images, depth_gt, image, args.max_tree_depth, pred_depths_r_list, \
                                     pred_depths_rc_list, pred_depths_instances_r_list, pred_depths_instances_rc_list, num_semantic_classes, \
                                     instances, segmentation_map, labels, pred_depths_c_list, uncertainty_maps_list, pred_depths_u_list, unc_decoder)

            # Evaluate #
            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()

                eval_measures, unc_error = online_eval(args, model, dataloader_eval, gpu, epoch, ngpus_per_node, group,
                                                       original_d3vo=args.d3vo_original)

                if eval_measures is not None:
                    if args.unc_head:
                        if best_unc > unc_error:

                            old_best_step = best_eval_steps[9]
                            model_path = '/model-{}-best_{}_{:.5f}'.format(old_best_step, "unc_d3vo", unc_error)
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)

                            best_unc = unc_error
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, "unc_d3vo", best_unc)

                            print('New best for {}. Saving model: {}'.format("unc_d3vo", model_save_name))
                            checkpoint = {
                                'global_step': global_step,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                'best_eval_steps': best_eval_steps
                            }

                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                            best_eval_steps[9] = global_step

                    exp_name = args.exp_name
                    log_txt = os.path.join(args.log_directory + '/' + args.model_name, exp_name + '_logs.txt')

                    # Log eval measures to file #
                    with open(log_txt, 'a') as txtfile:
                        txtfile.write(">>>>>>>>>>>>>>>>>>>>>>>>>Step:%d>>>>>>>>>>>>>>>>>>>>>>>>>\n" %
                                      (int(global_step)))
                        txtfile.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format(
                            'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
                        txtfile.write("depth estimation\n")
                        line = ''
                        for i in range(9):
                            line += '{:7.4f}, '.format(eval_measures[i])
                        txtfile.write(line + '\n')

                    # Update best #
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                            old_best = best_eval_measures_higher_better[i - 6].item()
                            best_eval_measures_higher_better[i - 6] = measure.item()
                            is_best = True

                        # New best - save checkpoint #
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
                            checkpoint = {
                                'global_step': global_step,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                'best_eval_steps': best_eval_steps
                            }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

                    eval_summary_writer.flush()
                    print("Best evals at global step: " + str(global_step))
                    print(best_eval_measures_higher_better.cpu().numpy())
                    print(best_eval_measures_lower_better.cpu().numpy())
                    if args.unc_head:
                        print("Best eval for uncertainty decoder: " + str(best_unc))

                model.train()
                if args.unc_head:  # Set some layers to eval to train the uncertainty decoder
                    model.module.set_to_eval_unc()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()

    print("Training ended with the following best evals: \n")
    print(best_eval_measures_higher_better)
    print(best_eval_measures_lower_better)
    if args.unc_head:
        print("Best eval for uncertainty decoder: " + str(best_unc))


def main():
    torch.cuda.empty_cache()
    gc.collect()

    # Create log dirs #
    exp_name = args.exp_name
    args.log_directory = os.path.join(args.log_directory, exp_name)
    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    # Save config file #
    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    # Save source files #
    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')

        command = 'cp train.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print(
            "This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'"
        )
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics.".
              format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
