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
                             
parser = argparse.ArgumentParser(description='PyTorch MOT Training')
parser.add_argument('--config', default='./experiments/config_search.yaml')
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
parser.add_argument('--epochs',type=int,default=5,help='num of training epochs')
parser.add_argument('--init_channels_image',type=int,default=64,help='num of init channels')
parser.add_argument('--init_channels_lidar',type=int,default=64,help='num of init channels')
parser.add_argument('--out_channels',type=int,default=512,help='num of init channels')
parser.add_argument('--layers',type=int,default=2,help='total number of layers')
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
parser.add_argument('--search_mode', type=int, default=2, help='0-search image branch, 1-lidar branch, 2-all-branch')

global args
args = parser.parse_args()

args.save = './experiments/search_exp/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
search_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
print("Using search mode {}".format(args.search_mode))
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
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    torch.manual_seed(args.seed)
    cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
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
        layers_lidar = 2,
        search_mode= args.search_mode,
        score_arch=config.model.score_arch,
        softmax_mode=config.model.softmax_mode,
        test_mode=config.model.test_mode,
        affinity_op=config.model.affinity_op,
        end_arch=config.model.end_arch,
        end_mode=config.model.end_mode, 
        neg_threshold=config.model.neg_threshold,
        criterion = criterion)
    model = model.cuda()
    logging.info("param size = %fMB", search_utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = build_optim(model, config)

    #train_transform, valid_transform = search_utils._data_transforms_cifar10(args)
    train_transform, valid_transform = build_augmentation(config.augmentation)
    # train_data = dset.CIFAR10(root=args.data,
    #                           train=True,
    #                           download=True,
    #                           transform=train_transform)
    train_dataset = build_dataset(
        config,
        set_source='train',
        evaluate=False,
        train_transform=train_transform)

    train_sampler = DistributedGivenIterationSampler(
        train_dataset,
        config.lr_scheduler.max_iter,
        config.batch_size,
        world_size=1,
        rank=0,
        last_iter=-1)

    train_queue = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler)

    valid_dataset = build_dataset(
        config,
        set_source='train',
        evaluate=False,
        train_transform=train_transform)

    valid_sampler = DistributedGivenIterationSampler(
        valid_dataset,
        config.lr_scheduler.max_iter * args.epochs,
        config.batch_size,
        world_size=1,
        rank=0,
        last_iter=-1)
    #print("train_queue", len(train_queue))
    num_train = config.lr_scheduler.max_iter
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    valid_queue = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=valid_sampler)
    
    valid_queue = iter(valid_queue)

    #torch.utils.data.sampler.SubsetRandomSampler(indices[0:num_train])
    val_dataset = build_dataset(
        config,
        set_source='val',
        evaluate=True,
        valid_transform=valid_transform)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    #scheduler = build_lr_scheduler(config.lr_scheduler, optimizer)
    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train(train_queue, valid_queue, model,
              architect, criterion, optimizer, lr)
        # validation
        MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = validate(val_dataset, model, str(epoch + 1), part = "val")

        #search_utils.save(model, os.path.join(args.save, 'weights.pt'))
        save_checkpoint(
                {
                    'epoch': epoch,
                    'score_arch': config.model.score_arch,
                    'appear_arch': config.model.appear_arch,
                    'best_mota': MOTA,
                    'state_dict': model.state_dict(),
                }, True, config.save_path + '/ckpt')

def train(train_queue, valid_queue, model, architect, criterion, optimizer,
          lr):    
    batch_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    losses = AverageMeter(config.print_freq)

    
    model.train()
    logger = logging.getLogger('global_logger')
    print("length", len(train_queue))
    step = 0

    end = time.time()
    for i, (input, det_info, det_id, det_cls, det_split) in enumerate(train_queue):
        step += 1
        n = input.size(0)

        #input = Variable(input, requires_grad=False).cuda()
        #target = Variable(target, requires_grad=False).cuda()
        input = Variable(input, requires_grad=False).cuda()
        if len(det_info) > 0:
            for k, v in det_info.items():
                det_info[k] = Variable(det_info[k],requires_grad=False).cuda() if not isinstance(det_info[k], list) else det_info[k]
        # print("valid_queue", len(valid_queue))
        input_search, det_info_search, det_id_search, det_cls_search, det_split_search = valid_queue.next()
        input_search = Variable(input_search, requires_grad=False).cuda()
        if len(det_info_search) > 0:
            for k, v in det_info_search.items():
                det_info_search[k] = Variable(det_info_search[k],requires_grad=False).cuda() if not isinstance(det_info_search[k], list) else det_info_search[k]
        data_time.update(time.time() - end)

        architect.step(input.squeeze(0), 
                       det_info, 
                       det_id, 
                       det_cls, 
                       det_split,
                       input_search.squeeze(0), 
                       det_info_search, 
                       det_id_search, 
                       det_cls_search, 
                       det_split_search,
                       lr,
                       optimizer,
                       unrolled=args.unrolled)

        optimizer.zero_grad()
        loss = model._loss(input.squeeze(0), 
                            det_info, 
                            det_id, 
                            det_cls, 
                            det_split)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.item())
        if (step + 1) % config.print_freq == 0:
            logger.info('Iter: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            step + 1,
                            len(train_queue),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses))
        end = time.time()

def validate(val_loader,
             model,
             step,
             part='train'):

    logger = logging.getLogger('global_logger')
    for i, (sequence) in enumerate(val_loader):
        logger.info('Test: [{}/{}]\tSequence ID: KITTI-{}'.format(
            i, len(val_loader), sequence.name))
        seq_loader = DataLoader(
            sequence,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True)
        if len(seq_loader) == 0:
            model.set_eval()
            logger.info('Empty Sequence ID: KITTI-{}, skip'.format(
                sequence.name))
        else:
            validate_seq(seq_loader, model)

        write_kitti_result(
            args.result_path,
            sequence.name,
            step,
            model.frames_id,
            model.frames_det,
            part=part)
    MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = evaluate(step, args.result_path, part=part)

    model.set_train()
    return MOTA, MOTP, recall, prec, F1, fp, fn, id_switches


def validate_seq(val_loader,
                 model,
                 fusion_list=None,
                 fuse_prob=False):
    batch_time = AverageMeter(0)

    # switch to evaluate mode
    model.set_eval()

    logger = logging.getLogger('global_logger')
    

    with torch.no_grad():
        for i, (input, det_info, dets, det_split) in enumerate(val_loader):
            input = input.cuda()
            if len(det_info) > 0:
                for k, v in det_info.items():
                    det_info[k] = det_info[k].cuda() if not isinstance(
                        det_info[k], list) else det_info[k]
            start = time.time()
            # compute output
            aligned_ids, aligned_dets, frame_start = model.predict(
                input[0], det_info, dets, det_split)
            end = time.time()
            batch_time.update(end - start)
            
            if i % config.print_freq == 0:
                logger.info(
                    'Test Frame: [{0}/{1}]\tTime '
                    '{batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time))




if __name__ == '__main__':
    main()
