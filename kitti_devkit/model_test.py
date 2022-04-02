import os
import sys
import time
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import search_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import yaml
from kitti_devkit.evaluate_tracking import evaluate
from torch.autograd import Variable
from tracking_model_search import TrackingNetwork
from architect import Architect
import search_utils
from easydict import EasyDict
from utils.data_util import write_kitti_result
from utils.build_util import (build_augmentation, build_criterion,
                              build_dataset, build_lr_scheduler, build_model,
                              build_optim)
from utils.train_util import (AverageMeter, DistributedGivenIterationSampler,
                              create_logger, load_state, load_partial_state, load_part_model,save_checkpoint)
                             
parser = argparse.ArgumentParser(description='PyTorch mmMOT Training')
parser.add_argument('--config', default='./experiments/config.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--result-path', default='./experiments/resutls', type=str)

parser.add_argument('--data',type=str,default='../data',help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.025,help='init learning rate')
parser.add_argument('--learning_rate_min',type=float,default=0.001,help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay',type=float,default=3e-4,help='weight decay')
parser.add_argument('--report_freq',type=float,default=50,help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs',type=int,default=50,help='num of training epochs')
parser.add_argument('--init_channels_image',type=int,default=64,help='num of init channels')
parser.add_argument('--init_channels_lidar',type=int,default=64,help='num of init channels')
parser.add_argument('--out_channels',type=int,default=512,help='num of init channels')
parser.add_argument('--layers',type=int,default=5,help='total number of layers')
parser.add_argument('--model_path',type=str,default='saved_models',help='path to save the model')
parser.add_argument('--cutout',action='store_true',default=False,help='use cutout')
parser.add_argument('--cutout_length',type=int,default=16, help='cutout length')
parser.add_argument('--drop_path_prob',type=float, default=0.3,help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip',type=float,default=5,help='gradient clipping')
parser.add_argument('--train_portion',type=float,default=0.5,help='portion of training data')
parser.add_argument('--unrolled', action='store_true',default=False,help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate',type=float,default=3e-4,help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',type=float,default=1e-3,help='weight decay for arch encoding')
global args
args = parser.parse_args()

args.save = './experiments/search_exp/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
search_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    global config, best_mota

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    torch.manual_seed(args.seed)
    cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    
    # criterion = nn.CrossEntropyLoss()
    criterion = build_criterion(config.loss)
    criterion = criterion.cuda()

    # model = TrackingNetwork(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = TrackingNetwork(
        C_image = args.init_channels_image,
        C_lidar = args.init_channels_image,
        out_channels = args.out_channels,
        layers_image = 2,
        layers_lidar = 1,
        seq_len=config.sample_max_len,
        score_arch=config.model.score_arch,
        softmax_mode=config.model.softmax_mode,
        test_mode=config.model.test_mode,
        affinity_op=config.model.affinity_op,
        end_arch=config.model.end_arch,
        end_mode=config.model.end_mode, 
        without_reflectivity=config.without_reflectivity,
        neg_threshold=config.model.neg_threshold,
        criterion = criterion)
    model = model.cuda()

    model.train()
    print("done")


if __name__ == '__main__':
    main()