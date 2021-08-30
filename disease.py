from __future__ import print_function
from os.path import exists, join, basename
from os import makedirs, remove
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from pylab import rcParams
import torch.nn.functional as F
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch jun')
parser.add_argument('--test_folder', type=str, default='./test', help='input image to use')
parser.add_argument('--model', type=str, default='./model/model_epoch_500.pth', help='model file to use')
parser.add_argument('--save_folder', type=str, default='./test', help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda', default='true')

opt = parser.parse_args()

print(opt)


def centeredCrop(img):
    width, height = img.size  # Get dimensions
    new_width = width - width % 8
    new_height = height - height % 8
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img


def main():
    images_list1 = glob('mri-010.jpg')
    images_list2 = glob('ct-010.jpg')
    name1 = []
    name2 = []
    model = torch.load(opt.model)
    index = 0
    if opt.cuda:
        model = model.cuda()
    for i, image_path in enumerate(images_list1):
        name1.append(image_path)
    for i, image_path in enumerate(images_list2):
        name2.append(image_path)

    for i in enumerate(images_list1):
        img1 = Image.open(name1[index]).convert('YCbCr')
        img0 = Image.open(name2[index]).convert('YCbCr')
        img1 = centeredCrop(img1)
        img0 = centeredCrop(img0)
        y1, cb1, cr1 = img1.split()
        y0, cb0, cr0 = img0.split()
        LR1 = y1
        LR0 = y0
        LR1 = Variable(ToTensor()(LR1)).view(1, -1, LR1.size[1], LR1.size[0])
        LR0 = Variable(ToTensor()(LR0)).view(1, -1, LR0.size[1], LR0.size[0])
        if opt.cuda:
            LR1 = LR1.cuda()
            LR0 = LR0.cuda()
        with torch.no_grad():
            tem1 = model.Extraction(LR1)
            tem0 = model.Extraction(LR0)
            tem1 = torch.exp(tem1)
            tem0 = torch.exp(tem0)
            #k1 = tem1 ** 2 / (tem0 ** 2 + tem1 ** 2)
            #k0 = tem0 ** 2 / (tem0 ** 2 + tem1 ** 2)
            k1 = tem1 / (tem0 + tem1)
            k0 = tem0 / (tem0 + tem1)

            #k1 = abs(tm1**1) / (abs(tm0**1) + abs(tm1**1))
            #k0 = abs(tm0**1) / (abs(tm0**1) + abs(tm1**1))
            tem = tem1 * k1 + tem0 * k0
            #tem = tem1 + tem0
            tem = model.Reconstruction(tem)
            tem = tem.cpu()
            tem = process(tem, cb0, cr0)
            misc.imsave(name2[index], tem)
            index += 1




if __name__ == '__main__':
    main()