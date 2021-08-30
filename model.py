import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import math
import torchvision.models as models

class FeatureExtraction(nn.Module):
    def __init__(self,  level):
        super(FeatureExtraction, self).__init__()
        self.level = level
        self.conv1 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.down = nn.AvgPool2d(2, 2)
        #self.conv0 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        #self.lu = nn.ReLU()

        self.block =block()
        self.level = level
    def forward(self, x):
        tem = self.conv1(x)
        #print("x:", x.shape)
        #a = torch.mul(self.lu(self.up(self.down(x))), tem) + x
        #tem = self.block(tem)

 #       for i in range(self.level):

        tem = self.block(tem)
        #return torch.mul(self.lu(self.up(self.down(a))), tem) + a
        return tem



class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))



        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.down = nn.AvgPool2d(2, 2)
        self.norm = nn.BatchNorm2d(64)
        self.lu = nn.ReLU()

    def forward(self, x):
        # print("x:", x.shape)
        out1 = self.conv1(x)
        # out2 = self.conv1(out1)
        out2 = self.norm(out1)
        out2 = self.lu(out2)
        # print("out2:", out2.shape)
        out3 = self.conv2(out2)
        # out4 = self.conv2(out3)
        out4 = self.norm(out3)
        out4 = self.lu(out4)
        # print("out4:", out4.shape)
        # out5 = self.conv3(out4)
        out6 = self.conv3(out4)
        out6 = self.norm(out6)
        out6 = self.lu(out6)

        out7 = self.conv2(self.conv2(x)) + x
        out7 = self.norm(out7)
        out7 = self.lu(out7)
        out8 = self.conv2(self.conv2(out7)) + out7
        out8 = self.norm(out8)
        out8 = self.lu(out8)
        out9 = self.conv2(self.conv2(out8)) + out8
        out9 = self.norm(out9)
        out9 = self.lu(out9)

        output1 = out2 + out4 + out6 + out9
        output1 = torch.mul(self.lu(x), output1) + x
        output1 = self.norm(output1)
        # print("output1:", output1.shape)
        return output1

        # out6 = self.conv0(output1)
        out10 = self.conv1(output1)
        # out11 = self.conv1(out10)
        out11 = self.norm(out10)
        out11 = self.lu(out11)

        out12 = self.conv2(out11)
        # out13 = self.conv2(out12)
        out13 = self.norm(out12)
        out13 = self.lu(out13)

        out14 = self.conv3(out13)
        # out15 = self.conv3(out14)
        out15 = self.norm(out14)
        out15 = self.lu(out15)

        out16 = self.conv2(self.conv2(output1)) + output1
        out16 = self.norm(out16)
        out16 = self.lu(out16)
        out17 = self.conv2(self.conv2(out16)) + out16
        out17 = self.norm(out17)
        out17 = self.lu(out17)
        out18 = self.conv2(self.conv2(out17)) + out17
        out18 = self.norm(out18)
        out18 = self.lu(out18)

        output2 = out11 + out13 + out15 + out18
        output2 = torch.mul(self.lu(output1), output2) + output1
        output2 = self.norm(output2)
        return output2


        # out12 = self.conv0(output2)
        out19 = self.conv1(output2)
        # out20 = self.conv1(out19)
        out20 = self.norm(out19)
        out20 = self.lu(out20)

        out21 = self.conv2(out20)
        # out22 = self.conv2(out21)
        out22 = self.norm(out21)
        out22 = self.lu(out22)

        out23 = self.conv3(out22)
        # out24 = self.conv3(out23)
        out24 = self.norm(out23)
        out24 = self.lu(out24)

        out25 = self.conv2(self.lu(self.conv2(output2))) + output2
        out25 = self.norm(out25)
        out25 = self.lu(out25)
        out26 = self.conv2(self.lu(self.conv2(out16))) + out25
        out26 = self.norm(out26)
        out26 = self.lu(out26)
        out27 = self.conv2(self.lu(self.conv2(out17))) + out26
        out27 = self.norm(out27)
        out27 = self.lu(out27)

        output3 = out20 + out22 + out24 + out27
        output3 = torch.mul(self.lu(output2), output3) + output2 + x

        output3 = self.norm(output3)
        return output3


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
        #self.conv3 = nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1))
        #self.conv2 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))



    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        #out3 = self.conv3(out2)
        return out2



class SRN(nn.Module):
    def __init__(self):
        super(SRN, self).__init__()
        self.Extraction = FeatureExtraction(level=3)
        self.Reconstruction = ImageReconstruction()

    def forward(self, input):
        tmp = self.Extraction(input)
        img = self.Reconstruction(tmp)

        return img

