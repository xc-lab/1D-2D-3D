import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import copy
import random
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import MetaDataset, FCNDataset
from models import Mnist, AlexNet, FCN


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training: Parkinson diagnose')
    parser.add_argument('--net-model', metavar='N', type=str, default='AlexNet',
                        help='classification model', dest='model', choices=['Mnist', 'lstmcnn', 'FCN'])
    parser.add_argument('--data', metavar='DIR', type=str, default='./data/training_data/',
                        help='path to training dataset')

    # configurations of the network
    parser.add_argument('--start-epoch', metavar='N', type=int, default=0,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', metavar='N', type=int, default=80,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', metavar='N', type=int, default=8,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--weighted-loss', type=bool, default=False,
                        help='Path of checkpoints')
    parser.add_argument('-j', '--workers', metavar='N', type=int, default=4,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # checkpoint and resume    #./checkpoints/best_results/model_LSTMCNN_best_X128.pth.tar
    parser.add_argument('--resume', metavar='PATH', type=str, default='',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--checkpoints', metavar='C', type=str, nargs='?', default='./checkpoints',
                        help='Path of checkpoints', dest='checkpoints')

    # chose optimizer for training
    parser.add_argument('--opt', default='Adam', choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'Adamax', 'Adadelta', 'ASGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', metavar='LR', type=float, default=1e-5,
                        help='initial learning rate')

    # weight decay for Adam optimizer
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    # schedulerfor SGD optimizer
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--milestones', type=str, default='1,2')
    parser.add_argument('--gamma', type=float, default=2 / 3)
    parser.add_argument('--early_stopping', metavar='N', type=int, default=-1, help='early stopping (default: -1)')

    # large batch size
    parser.add_argument('--large-batch-size', metavar='N', type=bool, default=False,
                        help='using large batch size (default: False)')
    parser.add_argument('--accumulation_steps', metavar='N', type=int, default=10,
                        help='The accumulation steps of using large batch size (default: 10)')

    args = parser.parse_args()
    print(args)


    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    writer = SummaryWriter(os.path.join('runs', args.model, current_time))


    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    if not os.path.exists(os.path.join(args.checkpoints, args.model)):
        os.mkdir(os.path.join(args.checkpoints, args.model))
    best_dir = os.path.join(args.checkpoints, args.model, current_time)
    os.makedirs(best_dir)



    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 200)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if args.weighted_loss and torch.cuda.is_available():
        class_weight = torch.FloatTensor([0.5, 0.5]).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight).cuda()
    elif args.weighted_loss:
        class_weight = torch.FloatTensor([0.5, 0.5])
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    elif torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()



    if args.model == 'MNIST' or args.model == 'Mnist':
        model = Mnist(in_channels=1, n_classes=2)
    elif args.model == 'alexnet' or args.model == 'AlexNet':
        model = AlexNet()
    elif args.model == 'FCN' or args.model == 'fcn':
        model = FCN()
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    print('All Parameters:%s %d'%(args.model, sum([torch.numel(param) for param in model.parameters()])))



    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError



    if args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['pred_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    # run function between the code line where uses GPU
    transform = transforms.Compose([transforms.ToTensor()])
    if args.model == 'FCN':
        train_dataset = FCNDataset(root=args.data, train=True, transform=transform)
    else:
        train_dataset = MetaDataset(root=args.data, train=True, transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    if args.model == 'FCN' :
        val_dataset = FCNDataset(root=args.data, train=False, transform=transform)
    else:
        val_dataset = MetaDataset(root=args.data, train=False, transform=transform)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)


    best_acc = 0
    best_loss = 1.0
    accumulation_steps = 10
    preds, targets = list(), list()
    best_weights  = copy.deepcopy(model.state_dict())
    for epoch in range(args.start_epoch, args.epochs):
        train_bar = tqdm(train_dataloader)
        running_results = {'total_data_size': 0, 'loss': 0, 'acc': 0}

        model.train()
        for i, data in enumerate(train_bar):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            _, pred_labels = outputs.max(1)
            pred = pred_labels.eq(labels).sum().item()
            acc = pred / args.batch_size

            if not args.large_batch_size:
                # compute gradient and do SGD step
                model.zero_grad()
                loss.backward()
                optimizer.step()

                running_results['total_data_size'] += args.batch_size
                running_results['loss'] += loss.item() * args.batch_size
                running_results['acc'] += acc * args.batch_size

                train_bar.set_description(desc='Epoch[%003d/%d]: Training, loss: %.4f, acc: %.4f' % (epoch + 1, args.epochs,
                                                                                                     running_results['loss'] /
                                                                                                     running_results['total_data_size'],
                                                                                                     running_results['acc'] /
                                                                                                     running_results['total_data_size']))
            else:
                loss = loss / accumulation_steps  # Normalize our loss (if averaged)
                loss.backward()  # Backward pass
                if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
                    optimizer.step()  # Now we can do an optimizer step
                    model.zero_grad()  # Initialize gradient with all 0 for next step

                    running_results['total_data_size'] += args.batch_size * accumulation_steps
                    running_results['loss'] += loss.item() * args.batch_size * accumulation_steps
                    running_results['acc'] += acc * args.batch_size * accumulation_steps

                    train_bar.set_description(desc='Epoch[%003d/%d]: Train, loss = %.4f, acc = %.4f' % (epoch + 1, args.epochs,
                                                                                  running_results['loss'] /
                                                                                  running_results['total_data_size'],
                                                                                  running_results['acc'] /
                                                                                  running_results['total_data_size']))
            # writer.add_scalar('Train/loss', loss, epoch * len(train_dataloader) + i)
            # writer.add_scalar('Train/acc', acc, epoch * len(train_dataloader) + i)

        train_epoch_loss = running_results['loss']/running_results['total_data_size']
        train_epoch_acc  = running_results['acc']/running_results['total_data_size']
        writer.add_scalar('Train/loss',  train_epoch_loss,  epoch + 1)
        writer.add_scalar('Train/acc',   train_epoch_acc,   epoch + 1)

        model.eval()
        with torch.no_grad():

            val_bar = tqdm(val_dataloader)
            val_results = {'total_data_size': 0, 'acc': 0, 'loss': 0}

            for j, data in enumerate(val_bar):
                inputs, targets = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, pred_labels = outputs.max(1)
                pred = pred_labels.eq(targets).sum().item()
                acc = pred / args.batch_size

                val_results['total_data_size'] += args.batch_size
                val_results['loss'] += loss.item() * args.batch_size
                val_results['acc'] += acc * args.batch_size


                val_bar.set_description(desc='                Test, loss = %.4f, acc = %.4f' % (
                val_results['loss'] / val_results['total_data_size'],
                val_results['acc'] / val_results['total_data_size']))

                # writer.add_scalar('Validation/loss', loss, epoch * len(val_dataloader) + j)
                # writer.add_scalar('Validation/acc',  acc,  epoch * len(val_dataloader) + j)

            val_epoch_loss = val_results['loss']/val_results['total_data_size']
            val_epoch_acc  = val_results['acc']/val_results['total_data_size']
            writer.add_scalar('Validation/loss', val_epoch_loss, epoch + 1)
            writer.add_scalar('Validation/acc',  val_epoch_acc,  epoch + 1)


            if val_epoch_acc > best_acc and val_epoch_loss < best_loss:
            # if val_epoch_loss < best_loss:
                best_epoch = epoch + 1

                best_acc = val_epoch_acc
                best_loss = val_epoch_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'pred_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(best_dir, 'model_'+ args.model +'_best_X128.pth.tar'))
                print('better epoch: {}, better_loss: {:.2f}'.format(best_epoch, best_acc))

    print('\n best epoch: {}, best_loss: {:.2f}'.format(best_epoch, best_acc))

if __name__=='__main__':
    main()