import os
import sys
import time
import gc

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

import numpy as np

from dataloaders.dataloader import DataLoaderCustom
from networks.NewCRFDepth import NewCRFDepth
from parser_options import train_parser
from custom_logging import debug_result, debug_visualize_gt_instances, tb_visualization, tb_visualization_unc
from online_eval import online_eval
from utils import silog_loss, l1_loss, unc_loss, \
                    eval_metrics, block_print, enable_print, load_checkpoint_skip_update_project, load_checkpoint, \
                    set_hparams_dict, sigma_metric_from_canonical_and_scale, sigma_metric_from_canonical_and_scale_nddepth


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
                        segmentation_active=args.segmentation,  concat_masks=args.concat_masks, instances_active=args.instances, \
                        roi_align=args.roi_align, roi_align_size=args.roi_align_size, \
                        bins_scale=args.bins_scale, unc_head=args.unc_head, virtual_depth_variation=args.virtual_depth_variation, \
                        upsample_type=args.upsample_type, bins_type=args.bins_type, bins_type_scale=args.bins_type_scale, unc_loss_type=args.unc_loss_type)
    model.train()
    # Set some layers to eval to train the uncertainty decoder #
    if args.unc_head:  
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
        # Single process, one machine #
        print("== Model Initialized (DataParallel)")
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = torch.optim.Adam([{'params': model.module.parameters()}], lr=args.learning_rate)

    # Load checkpoint and print stats #
    if args.unc_head:
        load_checkpoint(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)
   
     # IEBINS load checkpoint - skip some layers #
    else:  
        # IEBINS saved models - these are used for scale variations #
        if 'saved_models' in args.checkpoint_path:  
            load_checkpoint_skip_update_project(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)
        else:
            load_checkpoint(args.checkpoint_path, args.gpu, args.retrain, model, optimizer)

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)
    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum / var_cnt))

    cudnn.benchmark = True

    # TB logging #
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
        hparams = set_hparams_dict(args, num_semantic_classes, num_instances)
        writer.add_hparams(hparam_dict=hparams, metric_dict={})

    # Losses #
    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    l1_criterion = l1_loss()
    unc_criterion = unc_loss(unc_loss_type=args.unc_loss_type) # For uncertainty

    sigma_metric_from_canonical_and_scale_func = sigma_metric_from_canonical_and_scale if args.unc_loss_type != 2 else sigma_metric_from_canonical_and_scale_nddepth

    start_time = time.time()
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = 0
    global_step = 0
    duration = 0

    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(10, dtype=np.int32)
    best_unc = np.inf  
    
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate
    group = dist.new_group([i for i in range(ngpus_per_node)])

    # Initialize vars #
    pred_depths_r_list, pred_depths_rc_list, \
        pred_depths_c_list, uncertainty_maps_list, \
        pred_depths_u_list = [], [], [], [], []
    
    segmentation_map, instances, labels, unc_decoder = None, None, None, None
    model_just_loaded = True

    while epoch < args.num_epochs:
        if args.distributed: # Do proper shuffling
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()

            before_op_time = time.time()
            current_loss_depth = 0
            current_loss_unc_decoder = 0

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            b, _, h, w = depth_gt.shape
            num_images = image.shape[0]

            if args.segmentation:
                segmentation_map = torch.autograd.Variable(sample_batched['segmentation_map'].cuda(
                    args.gpu, non_blocking=True))

            if args.instances:
                instances = torch.autograd.Variable(sample_batched['instances_masks'].cuda(args.gpu, non_blocking=True))
                boxes = torch.autograd.Variable(sample_batched['instances_bbox'].cuda(args.gpu, non_blocking=True))
                labels = torch.autograd.Variable(sample_batched['instances_labels'].cuda(args.gpu, non_blocking=True))

            # Predict #
            if args.instances:
                result = model(image, masks=segmentation_map, instances=instances, boxes=boxes, labels=labels)
            elif args.segmentation:
                result = model(image, masks=segmentation_map)
            else:
                result = model(image, epoch, step)

            #debug_visualize_gt_instances(instances, mask, depth_gt)
            #debug_result(result, depth_gt)

            #################
            # Unpack result #
            #################
            pred_depths_r_list = result["pred_depths_r_list"]
            pred_depths_c_list = result["pred_depths_c_list"]
            uncertainty_maps_list = result["uncertainty_maps_list"]
            
            # Get scale and canonical #
            if args.update_block != 0:
                pred_scale_list = result["pred_scale_list"]
                pred_depths_rc_list = result["pred_depths_rc_list"]
            
            # Get uncertainties from extra heads #
            if args.unc_head:
                unc_c = result["unc_c"][-1]
                unc_s = result["unc_s"][-1]

            # gt_depth masking #
            mask = depth_gt > 0.1
            
            # Network is trained to optimize canonical and scale uncertainties #
            if args.unc_head:
                sigma_metric = sigma_metric_from_canonical_and_scale_func(pred_depths_rc_list[-1], unc_c,
                                                                     pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1),
                                                                     unc_s.unsqueeze(-1).unsqueeze(-1), args)
                if args.instances:
                    sigma_metric = torch.sum(sigma_metric * instances, dim=1).unsqueeze(1)
                    pred_d = torch.sum((pred_depths_r_list[-1] * instances), dim=1).unsqueeze(1)

                elif args.segmentation:
                    # [batch_size, num_semantic_classes, h, w]
                    sigma_metric = torch.sum(sigma_metric * segmentation_map, dim=1).unsqueeze(1)
                    
                    # [batch_size, 1, h, w]
                    pred_d = torch.sum((pred_depths_r_list[-1] * segmentation_map), dim=1).unsqueeze(1)
                else:
                    pred_d = pred_depths_r_list[-1]

                loss = unc_criterion.forward(pred_d, depth_gt, sigma_metric, mask.to(torch.bool))

            # Network is trained to optimize depth #
            else:
                for curr_tree_depth in range(args.max_tree_depth):
                    # Depth #
                    if args.instances:
                        pred_d = torch.sum((pred_depths_r_list[curr_tree_depth] * instances), dim=1).unsqueeze(1)
                    elif args.segmentation:
                        pred_d = torch.sum((pred_depths_r_list[curr_tree_depth] * segmentation_map), dim=1).unsqueeze(1)
                    else:
                        pred_d = pred_depths_r_list[curr_tree_depth]

                    if args.loss_type == 0:
                        current_loss_depth += silog_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))
                    else:
                        current_loss_depth += l1_criterion.forward(pred_d, depth_gt, mask.to(torch.bool))

                loss = current_loss_depth

            ############
            # Optimize #
            ############
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate -
                              end_learning_rate) * (1 - global_step / num_total_steps)**0.9 + end_learning_rate
                param_group['lr'] = current_lr
            optimizer.step()
            ###############

            ###############
            # Print stats #
            ###############
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if args.unc_head:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, unc_loss: {:.12f}'.format(epoch, \
                           step, steps_per_epoch, global_step, current_lr, loss))
                else:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, \
                           step, steps_per_epoch, global_step, current_lr, loss))

            duration += time.time() - before_op_time

            ##############
            # Logging TB #
            ##############
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)

                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0 # For the next loop

                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))

                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(
                    print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(),
                                        var_sum.item() / var_cnt, time_sofar, training_time_left))

                ###################
                # Tensorboard viz #
                ###################
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    if args.unc_head:
                        if args.unc_loss_type == 2:
                            # NDDepth [e-5, 1] with e-5 high uncert
                            # reverse it for viz
                            offset =  1 + np.exp(-5)
                            sigma_metric = offset - sigma_metric
                        tb_visualization_unc(writer, loss, sigma_metric, global_step, args, current_lr, var_sum, var_cnt,\
                                     num_images, depth_gt, image, args.max_tree_depth, pred_depths_r_list, \
                                     pred_depths_rc_list, num_semantic_classes, \
                                     instances, segmentation_map, labels, pred_depths_c_list, \
                                     uncertainty_maps_list, pred_depths_u_list)
                    else:
                        tb_visualization(writer, global_step, args, current_loss_depth, current_lr, var_sum, var_cnt,\
                                     num_images, depth_gt, image, args.max_tree_depth, pred_depths_r_list, \
                                     pred_depths_rc_list, num_semantic_classes, \
                                     instances, segmentation_map, labels, pred_depths_c_list, \
                                     uncertainty_maps_list, pred_depths_u_list)

            ############
            # Evaluate #
            ############
            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()

                eval_measures, unc_error = online_eval(args, model, dataloader_eval, gpu, epoch, ngpus_per_node, group,
                                                       unc_loss_type=args.unc_loss_type)
                if eval_measures is not None:
                    # Uncertainty metrics #
                    if args.unc_head:
                        if best_unc > unc_error:

                            old_best = best_eval_steps[9]
                            model_old_name = '/model-{}-best_{}_{:.5f}'.format(old_best, "unc", best_unc)
    
                            old_save_path = args.log_directory + '/' + args.model_name + model_old_name
                            if os.path.exists(old_save_path):
                                command = 'rm {}'.format(old_save_path)
                                os.system(command)

                            best_unc = unc_error
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, "unc", best_unc)

                            print('New best for {}. Saving model: {}'.format("unc", model_save_name))
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

                    # Depth metrics #
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
    
                    ##############################
                    # Print current overall best #
                    ##############################
                    print("Best evals at global step: " + str(global_step))
                    best_evals_list = best_eval_measures_lower_better.cpu().numpy().tolist(
                    ) + best_eval_measures_higher_better.cpu().numpy().tolist()
                    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
                        'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
                    for i in range(8):
                        print('{:7.4f}, '.format(best_evals_list[i]), end='')
                    print('{:7.4f}'.format(best_evals_list[-1]))

                    if args.unc_head:
                        print("Best eval for uncertainty: " + str(best_unc))

                model.train()
                # Set some layers to eval to train the uncertainty decoder #
                if args.unc_head:  
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
        print("Best eval for uncertainty: " + str(best_unc))


def main():
    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
    torch.cuda.empty_cache()
    gc.collect()

    # Create log dirs #
    args.log_directory = os.path.join(args.log_directory, args.exp_name)
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
