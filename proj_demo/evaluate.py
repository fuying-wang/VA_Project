#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

import models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(opt.data_dir, opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
else:
    if int(opt.ngpu) == 1:
        print('so we use gpu 1 for testing')
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        cudnn.benchmark = True
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

def test(test_video_loader,test_audio_loader,model,opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()

    sim_mat1 = np.zeros((30,30))
    sim_mat2 = np.zeros((30,30))
    right = 0
    for _, vfeat in enumerate(test_video_loader):
        for _, afeat in enumerate(test_audio_loader):
            bz = vfeat.size()[0]
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)
                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()
                cur_sim1 = model(vfeat_var, afeat_var)
                sim_mat1[:,k]=np.transpose(cur_sim1.cpu().data.numpy())

    sim_mat = sim_mat1
    np_indices = np.argsort(sim_mat,0)
    topk = np_indices[:opt.topk,:]
    right=0
    bz=opt.batchSize
    for k in np.arange(bz):
        order = topk[:,k]
        if k in order:
            right = right + 1
    print(torch.from_numpy(sim_mat))
    print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/bz))

def main():
    global opt
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))

    # create model
    model = models.VA_LSTM()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()

    test(test_video_loader, test_audio_loader, model, opt)

if __name__ == '__main__':
    main()

