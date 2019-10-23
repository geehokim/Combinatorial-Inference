from __future__ import print_function
import argparse
import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import datasets

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import configs.config as cf

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Network architecture
parser.add_argument('--net-type', default='resnet', type=str, help='model')

# Experiment options
parser.add_argument('--dataset', default='cub200', type=str)
parser.add_argument('--method', default='intranoisyset/baseline', type=str)
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--noise-rate', type=float, default=0.5, help='')
parser.add_argument('--noise-type', type=str, default='symmetric', help='')

# Miscs
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
valid_datasets = ['cub200']
assert args.dataset in valid_datasets, 'Invalid dataset'
assert args.noise_type in ['pairflip', 'symmetric']

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True


def main():
    exp_desc = 'pretrained' if args.pretrained else 'scratch'
    model_path = 'results/%s/%s/%s/%s/%s/noise_rate_%.2f/seed_%d' % (args.dataset, args.method, exp_desc, args.net_type,
                                                                     args.noise_type, args.noise_rate, args.seed)
    if not os.path.isdir(os.path.join(model_path, 'graphs')):
        mkdir_p(os.path.join(model_path, 'graphs'))

    # Data
    print('==> Preparing %s dataset' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize(int(cf.imresize[args.net_type])),
        transforms.RandomRotation(10),
        transforms.RandomCrop(cf.imsize[args.net_type]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(cf.imresize[args.net_type]),
        transforms.CenterCrop(cf.imsize[args.net_type]),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    if args.dataset == 'cub200':
        trainset = datasets.IntraNoisyCUB200(root='data/'+args.dataset, year=2011, train=True, download=True,
                                             transform=transform_train, noise_type=args.noise_type,
                                             noise_rate=args.noise_rate)
        testset = datasets.IntraNoisyCUB200(root='data/'+args.dataset, year=2011, train=False, download=False,
                                            transform=transform_test)
    else:
        assert False

    # Build dataloader
    train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,
                                   num_workers=args.workers, pin_memory=use_cuda, drop_last=True)
    test_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=use_cuda)

    # Construct the model
    print("==> creating model %s" % args.net_type)
    if args.net_type == 'resnet':
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, cf.num_classes[args.dataset])
    elif args.net_type == 'inception':
        model = models.inception_v3(pretrained=args.pretrained)
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, cf.num_classes[args.dataset])
    elif args.net_type == 'densenet':
        model = models.densenet161(pretrained=args.pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, cf.num_classes[args.dataset])
    elif args.net_type == 'vgg':
        model = models.vgg16(pretrained=args.pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         cf.num_classes[args.dataset])
    else:
        assert False

    model = model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    # Evaluation Only
    if args.evaluate:
        print('\nEvaluation only')
        checkpoint_path = os.path.join(model_path, 'checkpoint.pth.tar')
        assert os.path.isfile(checkpoint_path), 'Error: no checkpoint found!'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc = test(test_loader, model, criterion, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        return

    # Train model
    best_acc = 0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Open logger
    logger_path = os.path.join(model_path, 'log.txt')
    logger = Logger(logger_path)
    logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.'])

    # Train and test
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc = test(test_loader, model, criterion, use_cuda)

        # Logging for current iteration
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, dir=model_path, filename='checkpoint.pth.tar')

    logger.append([0, 0, 0, 0, best_acc])
    logger.close()
    print('Best acc:', best_acc)

    # Draw plots
    logger.plot(names=['Train Loss', 'Test Loss'])
    savefig(os.path.join(model_path, 'graphs/loss.png'))
    logger.plot(names=['Train Acc.', 'Test Acc.'])
    savefig(os.path.join(model_path, 'graphs/acc.png'))


def train(train_loader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(test_loader, model, criterion, use_cuda):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    return losses.avg, top1.avg


def save_checkpoint(state, dir, filename):
    filepath = os.path.join(dir, filename)
    torch.save(state, filepath)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()