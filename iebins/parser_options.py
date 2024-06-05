import argparse

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


# Train parser
train_parser = argparse.ArgumentParser(description='Scale-Aware SLAM PyTorch implementation.', fromfile_prefix_chars='@')
train_parser.convert_arg_line_to_args = convert_arg_line_to_args

train_parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
train_parser.add_argument('--model_name',                type=str,   help='model name', default='nyu')
train_parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='tiny07')
train_parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
train_parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
train_parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')

# Dataset
train_parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
train_parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
train_parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
train_parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
train_parser.add_argument('--input_height',              type=int,   help='input height', default=480)
train_parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
train_parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
train_parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=0.1)

# Arch 
train_parser.add_argument('--update_block',              type=int,   help='head block type', default='1')
train_parser.add_argument('--var',                       type=int,   help='variation of head block', default='0')
train_parser.add_argument('--padding_instances',         type=int,   help='how many pixels to padd the bbox', default='0')
train_parser.add_argument('--max_tree_depth',            type=int,   help='max GRU iterations', default='6')
train_parser.add_argument('--bin_num',                   type=int,   help='number of bins', default='16')
train_parser.add_argument('--d3vo',                      dest='d3vo',  help='predict uncertainty for scale d3vo', action='store_true')
train_parser.add_argument('--d3vo_c',                    dest='d3vo_c',  help='predict also the uncertainty for canonical depth with d3vo', action='store_true')
train_parser.add_argument('--d3vo_original',             dest='d3vo_original',  help='use original d3vo loss with no beta', action='store_true')
train_parser.add_argument('--segmentation',              dest='segmentation', help='segmentation variation', action='store_true')
train_parser.add_argument('--instances',                 dest='instances',    help='instances variation', action='store_true')
train_parser.add_argument('--roi_align',                 type=int,   help='use roi align', default='0')
train_parser.add_argument('--roi_align_size',            type=int,   help='size of roi align', default='32')
train_parser.add_argument('--bins_scale',                type=int,   help='Bins for scale', default='100')
train_parser.add_argument('--virtual_depth_variation',    type=int,   help='0 for scale/shift, for 1 for scale regression and 2 for scale with bins', default='1')

# Log and save
train_parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
train_parser.add_argument('--exp_name',                  type=str,   help='experiment name to save checkpoints and summaries', default='exp-1')
train_parser.add_argument('--log_freq',                  type=int,   help='logging frequency in global steps', default=100)
train_parser.add_argument('--save_freq',                 type=int,   help='checkpoint saving frequency in global steps', default=5000)

# Training hparams
train_parser.add_argument('--train_decoder',             type=int,   help='how many layers to train from the decoder', default=1)
train_parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
train_parser.add_argument('--loss_type',                 type=int,   help='0 for silog, 1 for l1', default=0)
train_parser.add_argument('--uncertainty_weight',        type=float, help='weight for the uncertainty loss', default=1), 
train_parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
train_parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
train_parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
train_parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
train_parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
train_parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
train_parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
train_parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
train_parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
train_parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
train_parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
train_parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
train_parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
train_parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
train_parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
train_parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
train_parser.add_argument('--multiprocessing_distributed',           help='use multi-processing distributed training to launch '
                                                                          'N processes per node, which has N GPUs. This is the '
                                                                          'fastest way to use PyTorch for either single node or '
                                                                          'multi node data parallel training', action='store_true',)
# Online eval
train_parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
train_parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
train_parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
train_parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
train_parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
train_parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
train_parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
train_parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
train_parser.add_argument('--eval_freq',                 type=int,   help='online evaluation frequency in global steps', default=500)
train_parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                          'if empty outputs to checkpoint folder', default='')


# Eval parser 
eval_parser = argparse.ArgumentParser(description='Scale-Aware SLAM PyTorch implementation.', fromfile_prefix_chars='@')
eval_parser.convert_arg_line_to_args = convert_arg_line_to_args

eval_parser.add_argument('--model_name',                type=str,   help='model name', default='iebins')
eval_parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='large07')
eval_parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
eval_parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
eval_parser.add_argument('--loss_type',                 type=int,   help='0 for silog, 1 for l1', default=0)


# Dataset
eval_parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
eval_parser.add_argument('--input_height',              type=int,   help='input height', default=480)
eval_parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
eval_parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
eval_parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=0)

# Arch
eval_parser.add_argument('--update_block',              type=int,   help='head block type', default='0')
eval_parser.add_argument('--var',                       type=int,   help='variation of head block', default='0')
eval_parser.add_argument('--max_tree_depth',            type=int,   help='max GRU iterations', default='6')
eval_parser.add_argument('--bin_num',                   type=int,   help='number of bins', default='16')
eval_parser.add_argument('--segmentation',              dest='segmentation', help='segmentation variation', action='store_true')
eval_parser.add_argument('--instances',                 dest='instances',    help='instances variation', action='store_true')
eval_parser.add_argument('--padding_instances',         type=int,            help='how many pixels to padd for bbox', default='0')
eval_parser.add_argument('--d3vo',                      help='d3vo uncertainty variation', action='store_true')
eval_parser.add_argument('--d3vo_c',                    help='d3vo uncertainty variation canonical', action='store_true')
eval_parser.add_argument('--bins_scale',                type=int,   help='Bins for scale', default='100')
eval_parser.add_argument('--virtual_depth_variation',   type=int,   help='0 for scale/shift, for 1 for scale regression and 2 for scale with bins', default='1')
eval_parser.add_argument('--roi_align',                 type=int,   help='use roi align', default='0')
eval_parser.add_argument('--roi_align_size',            type=int,   help='size of roi align', default='32')

# Preprocessing
eval_parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
eval_parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
eval_parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
eval_parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
eval_parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='.')
eval_parser.add_argument('--exp_name',                  type=str,   help='experiment name to save checkpoints and summaries', default='exp-1')
eval_parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
eval_parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
eval_parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
eval_parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
eval_parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
eval_parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
eval_parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
eval_parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
eval_parser.add_argument('--pick_class',                type=int,   help='evaluate single class for debug', default=0)


# Test parser 
test_parser = argparse.ArgumentParser(description='Scale-Aware SLAM PyTorch implementation - test', fromfile_prefix_chars='@')
test_parser.convert_arg_line_to_args = convert_arg_line_to_args

test_parser.add_argument('--model_name',                type=str,   help='model name', default='iebins')
test_parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='large07')
test_parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
test_parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
test_parser.add_argument('--loss_type',                 type=int,   help='0 for silog, 1 for l1', default=0)

# Dataset
test_parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
test_parser.add_argument('--input_height',              type=int,   help='input height', default=480)
test_parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
test_parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
test_parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=0)

# Arch
test_parser.add_argument('--update_block',              type=int,   help='head block type', default='0')
test_parser.add_argument('--var',                       type=int,   help='variation of head block', default='0')
test_parser.add_argument('--max_tree_depth',            type=int,   help='max GRU iterations', default='6')
test_parser.add_argument('--bin_num',                   type=int,   help='number of bins', default='16')
test_parser.add_argument('--segmentation',              dest='segmentation', help='segmentation variation', action='store_true')
test_parser.add_argument('--instances',                 dest='instances',    help='instances variation', action='store_true')
test_parser.add_argument('--padding_instances',         type=int,            help='how many pixels to padd for bbox', default='0')
test_parser.add_argument('--d3vo',                      help='d3vo uncertainty variation', action='store_true')
test_parser.add_argument('--d3vo_c',                    help='d3vo uncertainty variation canonical', action='store_true')
test_parser.add_argument('--bins_scale',                type=int,   help='Bins for scale', default='100')
test_parser.add_argument('--virtual_depth_variation',   type=int,   help='0 for scale/shift, for 1 for scale regression and 2 for scale with bins', default='1')
test_parser.add_argument('--roi_align',                 type=int,   help='use roi align', default='0')
test_parser.add_argument('--roi_align_size',            type=int,   help='size of roi align', default='32')

test_parser.add_argument('--save_dir',       type=str,   help='path to save predicted data', required=True)
test_parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=True)
test_parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=True)
test_parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=True)
test_parser.add_argument('--min_depth_test',            type=float, help='minimum depth for evaluation', default=1e-3)
test_parser.add_argument('--max_depth_test',            type=float, help='maximum depth for evaluation', default=80)
