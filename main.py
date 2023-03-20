import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src import test
# ddp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='UCRN',
                    help='name of the model')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for ddp training')
parser.add_argument('--ddp',  action='store_true', default=False, help='using distributedDataParallel')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--use_bert', type=bool, default=False, help='use bert text feature')
parser.add_argument('--train_mode', type=str, default='regression', help='classification or regression')
parser.add_argument('--data_path', type=str, default='UCRN/data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=3,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--temp_proj', type=int, default=1, help="temporal projection number after self-refinement")

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (SGD/Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

parser.add_argument('--pretrain', type=str, default=None, help='load pretrained model name')

parser.add_argument('--jsd', action='store_true', default=False, help="multimodal jsd loss")

# learning rate
parser.add_argument('--schedule', type=str, default='default', help='learning rate schedule; cycliclr or warmup or '
                                                                     'salr')
parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate for cyclicLR')
parser.add_argument('--max_lr', type=float, default=1e-2, help='max learning rate for cyclicLR')
parser.add_argument('--step_up', type=int, default=2000, help='step size up for cyclicLR on mosei, other dataset '
                                                              'subject to change')
# warm up learning rate --schedule = 'warmup'
parser.add_argument('--lr_stepper', type=str, default='steplr', help='stepper lr after warmup; explr')
parser.add_argument('--stepper_size', type=int, default='10', help='stepLR decay step size')
parser.add_argument('--warm_epoch', type=int, default='5', help='number of warmup epochs')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=581,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='UCRN',
                    help='name of the trial (default: "URCN")')
parser.add_argument('--visualize', type=bool, default=False, help='loss plot enable')
parser.add_argument('--vis_dir', type=str, default='UCRN/visfolder',
                    help='path for storing the dataset')

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

use_cuda = False

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        if args.ddp:
            dist.init_process_group(backend='nccl')
            print("using cuda with ddp...")
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("using cuda without ddp...")
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
# print(len(train_data))  # mosi:1284
# print(len(valid_data))  # mosi:229
# print(len(test_data))   # mosi:686
if args.ddp:
    train_sampler = DistributedSampler(train_data)
    valid_sampler = DistributedSampler(valid_data)
    test_sampler = DistributedSampler(test_data)
    nproc = torch.cuda.device_count()
    shuffle, drop_last = False, False

else:
    train_sampler = valid_sampler = test_sampler = None
    nproc = 1
    shuffle, drop_last = True, False

train_loader = DataLoader(train_data, batch_size=int(args.batch_size/nproc), shuffle=shuffle,
                          sampler=train_sampler, generator=torch.Generator(device='cuda'))
valid_loader = DataLoader(valid_data, batch_size=int(args.batch_size/nproc), shuffle=shuffle,
                          sampler=valid_sampler, drop_last=drop_last, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=int(args.batch_size/nproc), shuffle=shuffle,
                         sampler=test_sampler, drop_last=drop_last, generator=torch.Generator(device='cuda'))

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
# print('hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v:', hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v)
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
# print('hyp_params.l_len, hyp_params.a_len, hyp_params.v_len:', hyp_params.l_len, hyp_params.a_len, hyp_params.v_len)
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = 1
hyp_params.criterion = 'L1Loss'

##
if __name__ == '__main__':
    if hyp_params.visualize:
        test.test_initiate(hyp_params, train_loader, valid_loader, test_loader)
    else:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
