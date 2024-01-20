import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import argparse
import numpy as np
from tqdm import tqdm

from networks.NewCRFDepth import NewCRFDepth
from utils import post_process_depth, flip_lr, silog_loss, l1_loss, compute_errors, eval_metrics, entropy_loss, colormap, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, eval_parser

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = eval_parser.parse_args([arg_filename_with_prefix])
else:
    args = eval_parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader

def eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()

    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
   
    image_counter = 0
    eval_images = 2
    for step, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        image_counter += 1
        if image_counter == eval_images:
            break

        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            num_log_images = image.shape[0]
            gt_depth = eval_sample_batched['depth']

            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            # Predict #
            result = model(image)
            
            # Unpack #            
            pred_depths_r_list = result["pred_depths_r_list"]
            pred_depths_c_list = result["pred_depths_c_list"]
            uncertainty_maps_list = result["uncertainty_maps_list"]
            if args.update_block != 0:
                pred_depths_rc_list = result["pred_depths_rc_list"]
                pred_scale_list = result["pred_scale_list"] 
                pred_shift_list = result["pred_shift_list"] 
            if args.update_block == 2:
                pred_depths_u_list = result["pred_depths_u_list"]
            if args.predict_unc == True:
                unc = result["unc"]
            if args.predict_unc_d3vo == True:
                unc_d3vo = result["unc_d3vo"]

            debug(result, gt_depth)

            max_tree_depth = len(pred_depths_r_list)
            for i in range(num_log_images):
                if args.dataset == 'nyu':
                    writer.add_image('gt_depth/image/{}'.format(i), colormap(torch.where(gt_depth < 1e-3, gt_depth * 0 + 1e-3, gt_depth)[i, :, :, :].data), step)
                    writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, step)
                    writer.add_image('depth_r_est0/image/{}'.format(i), colormap(pred_depths_r_list[0][i, :, :, :].data), step)
                    writer.add_image('depth_r_est1/image/{}'.format(i), colormap(pred_depths_r_list[1][i, :, :, :].data), step)
                    writer.add_image('depth_r_est2/image/{}'.format(i), colormap(pred_depths_r_list[2][i, :, :, :].data), step)
                    writer.add_image('depth_r_est3/image/{}'.format(i), colormap(pred_depths_r_list[3][i, :, :, :].data), step)
                    writer.add_image('depth_r_est4/image/{}'.format(i), colormap(pred_depths_r_list[4][i, :, :, :].data), step)
                    writer.add_image('depth_r_est5/image/{}'.format(i), colormap(pred_depths_r_list[5][i, :, :, :].data), step)
                    writer.add_image('depth_c_est0/image/{}'.format(i), colormap(pred_depths_c_list[0][i, :, :, :].data), step)
                    writer.add_image('depth_c_est1/image/{}'.format(i), colormap(pred_depths_c_list[1][i, :, :, :].data), step)
                    writer.add_image('depth_c_est2/image/{}'.format(i), colormap(pred_depths_c_list[2][i, :, :, :].data), step)
                    writer.add_image('depth_c_est3/image/{}'.format(i), colormap(pred_depths_c_list[3][i, :, :, :].data), step)
                    writer.add_image('depth_c_est4/image/{}'.format(i), colormap(pred_depths_c_list[4][i, :, :, :].data), step)
                    writer.add_image('depth_c_est5/image/{}'.format(i), colormap(pred_depths_c_list[5][i, :, :, :].data), step)
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


                for ii in range(max_tree_depth):
                    writer.add_image('uncer_est{}/image/{}'.format(ii, i), colormap(torch.log10(uncertainty_maps_list[ii][i, :, :, :].data), name='magma'), step)


            max_tree_depth = len(pred_depths_r_list)
            if post_process:
                image_flipped = flip_lr(image)
                result = model(image_flipped)
                pred_depths_r_list_flipped  = result["pred_depths_r_list"]
                pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])

                pred_depth = pred_depth.cpu().numpy().squeeze()
            else:
                pred_depth = pred_depths_r_list[-1].cpu().numpy().squeeze()

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
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

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
           
        if True:
            print("GT")            
            print(np.max(gt_depth[valid_mask]))
            print(np.min(gt_depth[valid_mask])) 
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
def debug(result, gt_depth):
    if True: 
        # Debug #
        if True:
            print("depth")
            print(torch.max(result['pred_depths_r_list'][5][0, :, :, :]))
            print(torch.min(result['pred_depths_r_list'][5][0, :, :, :]))
        
        if True:
            print("canonical")
            print(torch.max(result['pred_depths_rc_list'][5][0, 0, :, :]))
            print(torch.min(result['pred_depths_rc_list'][5][0, 0, :, :]))
            print(torch.mean(result['pred_depths_rc_list'][5][0, 0, :, :]))
            print(torch.std(result['pred_depths_rc_list'][5][0, 0, :, :]))
        if True:
            print("uncertainty (std)")
            print(torch.max(result['uncertainty_maps_list'][5][0, 0, :, :]))
            print(torch.min(result['uncertainty_maps_list'][5][0, 0, :, :]))
            print(torch.mean(result['uncertainty_maps_list'][5][0, 0, :, :]))
            print(torch.std(result['uncertainty_maps_list'][5][0, 0, :, :]))            
        if True:
            print("scale")
            print(torch.max(result['pred_scale_list'][5][0, 0]))
            print(torch.min(result['pred_scale_list'][5][0, 0]))
            print(torch.mean(result['pred_scale_list'][5][0, 0]))
            print(torch.std(result['pred_scale_list'][5][0, 0]))  
            #print(torch.max(result['pred_scale_list'][5][0, 0, :, :]))
            #print(torch.min(result['pred_scale_list'][5][0, 0, :, :]))
            #print(torch.mean(result['pred_scale_list'][5][0, 0, :, :]))
            #print(torch.std(result['pred_scale_list'][5][0, 0, :, :]))            
        if True:
            print("shift")
            print(torch.max(result['pred_shift_list'][5][0, 0]))
            print(torch.min(result['pred_shift_list'][5][0, 0]))
            print(torch.mean(result['pred_shift_list'][5][0, 0]))
            print(torch.std(result['pred_shift_list'][5][0, 0]))       
            #print(torch.max(result['pred_shift_list'][5][0, 0, :, :]))
            #print(torch.min(result['pred_shift_list'][5][0, 0, :, :]))
            #print(torch.mean(result['pred_shift_list'][5][0, 0, :, :]))
            #print(torch.std(result['pred_shift_list'][5][0, 0, :, :]))
        if True:
            print("unc_d3vo")
            print(torch.max(result['unc_d3vo'][0, 0, :, :]))
            print(torch.min(result['unc_d3vo'][0, 0, :, :]))
            print(torch.mean(result['unc_d3vo'][0, 0, :, :]))
            print(torch.std(result['unc_d3vo'][0, 0, :, :]))
        if True:
            print("sample")
            print(result['pred_depths_r_list'][5][0, 0, 120, 701])
            print(result['unc_d3vo'][0, 0, 120, 701])
            print(gt_depth[0, 120, 701, 0])

def main_worker(args):

    # CRF model
    model = NewCRFDepth(version=args.encoder, max_depth=args.max_depth, max_tree_depth=args.max_tree_depth, bin_num=args.bin_num, 
                        min_depth=args.min_depth, update_block=args.update_block, loss_type=args.loss_type, 
                        pretrained=args.pretrain, predict_unc=args.predict_unc, predict_unc_d3vo=args.predict_unc_d3vo)
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

    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        eval_measures = eval(model, dataloader_eval, post_process=True)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()

    exp_name = args.exp_name  
    args.log_directory = os.path.join(args.log_directory,exp_name)  
    print(args.log_directory)
    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
