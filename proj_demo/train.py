#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils
from sklearn.cross_validation import train_test_split
parser = argparse.ArgumentParser(description=
        'Runs the training code')
parser.add_argument('-c','--cuda', help='cuda',type = bool)
parser.add_argument('-ng','--ngpu', help='numbers of gpu',type = int)
parser.add_argument('-gi','--gpu_id',help='the index of gpu')
parser.add_argument('-d','--data_dir', help='the folder name')
parser.add_argument('-f','--flist', help='file list')
parser.add_argument('-w','--workers', help='numbers of work thread',type = int)
parser.add_argument('-bz','--batchSize',help='the size of a batch',type = int)
parser.add_argument('-ms','--manualSeed', help='manualSeed',type = int)
parser.add_argument('-a','--adam', help='adam')
parser.add_argument('-l','--lr', help='learning rate decay',type = float)
parser.add_argument('-ld','--lr_decay',help='max epochs',type = float)
parser.add_argument('-lde','--lr_decay_epoch', help='epochs of learning rate decay',type =float)
parser.add_argument('-m','--momentum', help='optimizer momentum')
parser.add_argument('-wd','--weight_decay', help='weight decay',type = float)
parser.add_argument('-me','--max_epochs',help='max epochs',type = int)
parser.add_argument('-pf','--print_freq',help='print epochs',type =int)
parser.add_argument('-s','--epoch_save', help='save epochs',type = int)
parser.add_argument('-cf','--checkpoint_folder', help='folder name of saveing checkpoint')
parser.add_argument('-p','--prefix', help='prefix')
parser.add_argument('-mo','--model',help='model')
parser.add_argument('-g','--gradient_clip', help='gradient_clip')
parser.add_argument('-im','--init_model',help='initial model',type = str)
parser.set_defaults(cuda=True,ngpu=1,gpu_id='0',data_dir='../Train',flist='./filelists/train_filelist.txt',
                    workers=4,batchSize=32,manualSeed=None, adam=True,lr=5e-4,lr_decay=0.7,lr_decay_epoch=10,
                    momentum=0,weight_decay=1e-5,max_epochs=200,print_freq=40,epoch_save=10,checkpoint_folder='./checkpoints_Lstm',
                    prefix='VA_METRIC',model='',gradient_clip=0.01,init_model='')
args = parser.parse_args()
print(args)
if args.checkpoint_folder is None:
    args.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(args.checkpoint_folder):
    os.system('mkdir {0}'.format(args.checkpoint_folder))

train_dataset = dset(args.data_dir, flist=args.flist)

print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    torch.manual_seed(args.manualSeed)
else:
    if int(args.ngpu) == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(args.gpu_id))

        if args.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
            torch.cuda.manual_seed(args.manualSeed)
            cudnn.benchmark = True
print('Random Seed: {0}'.format(args.manualSeed))


def to_np(x):
    return x.data.cpu().numpy()

# training function for metric learning
def train(train_loader, valid_loader, model, criterion, optimizer, epoch):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()

    # training mode
    model.train()
    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('int32')
        target1 = torch.from_numpy(label1).float()

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2).float()

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target
        
        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if args.cuda:
            vfeat_var = vfeat_var.cuda()
            #vfeat1_var = vfeat1_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()


        # forward, backward optimize
        cur_sim = model(vfeat_var, afeat_var)
        loss = criterion(cur_sim, target_var)

        ##############################
        # update loss in the loss meter
        ##############################
        #losses.update(loss.data[0], vfeat0.size(0))

        ##############################
        # compute gradient and do zero_grad
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, args.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss:.4f} \t'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=loss.data[0])
            print(log_str)

# learning rate adjustment function
def LR_Policy(optimizer, init_lr, policy):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * policy

# main function for training the model
def main():
    # train data loader
    X_train,X_valid = train_test_split(train_dataset,test_size=0.03)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=args.batchSize,
                                     shuffle=True, num_workers=int(args.workers))
    valid_loader = torch.utils.data.DataLoader(X_valid, batch_size=128,
                                     shuffle=True, num_workers=int(args.workers))
    # create model
    model=models.VA_LSTM()

    if args.init_model != '':
        print('loading pretrained model from {0}'.format(args.init_model))
        model.load_state_dict(torch.load(args.init_model))

    # Contrastive Loss
    criterion = nn.BCELoss()

    if args.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: args.lr_decay ** ((epoch + 1) // args.lr_decay_epoch)   #poly policy
    for epoch in range(args.max_epochs):
        #################################
        # train for one epoch
        #################################
        train(train_loader,valid_loader, model, criterion, optimizer, epoch)
        LR_Policy(optimizer, args.lr, lambda_lr(epoch))      # adjust learning rate through poly policy

        ##################################
        # save checkpoint every 10 epochs
        ##################################
        if ((epoch+1) % args.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(args.checkpoint_folder, args.prefix, epoch+1)
            utils.save_checkpoint(model, path_checkpoint)

if __name__ == '__main__':
    main()
