from __future__ import print_function
import argparse
from math import log10
from os.path import exists, join, basename
from os import makedirs, remove

import numpy as np
import torch
import time
import cv2


import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import SRN
from data import get_training_set, get_test_set



#from model import myCNN


# Training settings 
parser = argparse.ArgumentParser(description='PyTorch jun')
parser.add_argument('--batchSize', type=int, default=5, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default='true')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--nIters', type=int, default=1, help='Number of iterations in epoch')
opt = parser.parse_args()
print(opt)

#gpu_device = "cuda:0"

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.testBatchSize, shuffle=False)


def CharbonnierLoss(predict, target):
    return torch.mean((predict-target)**2 + 1e-6) # epsilon=1e-3

print('===> Building model')
model = SRN()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#criterion = CharbonnierLoss()
criterion = nn.MSELoss()
print(model)
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

losses = []
testloss = []
def train(epoch):
    NITERS = opt.nIters
    avg_loss = 0
    avg_test = 0
    for i in range(NITERS):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR = Variable(batch[0])
            if cuda:
                LR = LR.cuda()
                #HR_2_target = HR_2_target.cuda()
                #HR_4_target = HR_4_target.cuda()
                #HR_8_target = HR_8_target.cuda()
            optimizer.zero_grad()
            HR = model(LR)
            loss = ((HR - LR)**2).mean()
            # loss2 = CharbonnierLoss(HR_4, HR_4_target)
            # loss3 = CharbonnierLoss(HR_8, HR_8_target)
            avg_loss += loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("===> Epoch {}, Loop{}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))

        avg_loss += epoch_loss / len(training_data_loader)
        avg_loss /= NITERS
        losses.append(avg_loss)
        epoch_loss = 0
        for batch in testing_data_loader:
            LR = Variable(batch[0])
            if cuda:
                LR = LR.cuda()
            HR = model(LR)
            loss = ((HR - LR) ** 2).mean()
            avg_test += loss
            epoch_loss += loss.item()
        avg_test += epoch_loss / len(testing_data_loader)
        avg_loss /= NITERS
        testloss.append(avg_loss)

lr=opt.lr
psnrs = []
def test():
    avg_psnr1 = 0
    avg_psnr2 = 0
    avg_psnr3 = 0
    for batch in testing_data_loader:
        LR = Variable(batch[0])
        if cuda:
            LR = LR.cuda()
            #HR_4_target = HR_4_target.cuda()
            #HR_8_target = HR_8_target.cuda()

        HR = model(LR)
        loss = ((HR - LR) ** 2).mean()
        # mse2 = criterion(HR_4, HR_4_target)
        # mse3 = criterion(HR_8, HR_8_target)
        psnr1 = 10 * log10(1 / loss.item())
        # psnr2 = 10 * log10(1 / mse2.item())
        # psnr3 = 10 * log10(1 / mse3.data[0])
        avg_psnr1 += psnr1
        psnrs.append(avg_psnr2 / len(testing_data_loader))
        # avg_psnr2 += psnr2
        # avg_psnr3 += psnr3
    print("===> Avg. PSNR1: {:.4f} dB".format(avg_psnr1 / len(testing_data_loader)))
    print("===> Avg. PSNR2: {:.4f} dB".format(avg_psnr2 / len(testing_data_loader)))
    print("===> Avg. PSNR3: {:.4f} dB".format(avg_psnr3 / len(testing_data_loader)))

for epoch in range(1, opt.epochs + 1):
    train(epoch)
    test()
    if epoch % 500 == 0:
        lr = lr/2
        print('new learning rate {}'.format(lr))
    model_out_path = "./model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
np.save('losses', losses)
np.save('testloss', testloss)
#np.save('psnrs', psnrs)
