from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
from .resnet import *
import torchvision


class PCB_model(nn.Module):
    def __init__(self, num_stripes=6, num_features=256, num_classes=0, norm=False, dropout=0, last_stride=1,
                 reduced_dim=256, share_conv=True, output_feature=None):
        super(PCB_model, self).__init__()
        # Create PCB_only model
        self.num_stripes = num_stripes
        self.num_features = num_features
        self.num_classes = num_classes
        self.rpp = False
        self.reduced_dim = reduced_dim
        self.output_feature = output_feature
        self.share_conv = share_conv

        # ResNet50: from 3*384*128 -> 2048*24*8 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])
        # decrease the downsampling rate
        if last_stride != 2:
            # decrease the downsampling rate
            # change the stride2 conv layer in self.layer4 to stride=1
            self.base[7][0].conv2.stride = last_stride
            # change the downsampling layer in self.layer4 to stride=1
            self.base[7][0].downsample[0].stride = last_stride

        self.dropout = dropout
        out_planes = 2048
        self.local_conv = nn.Conv2d(out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
        init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
        self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
        init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
        init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

##---------------------------stripe1----------------------------------------------#
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance0.weight, std=0.001)
        init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance1.weight, std=0.001)
        init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance2.weight, std=0.001)
        init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance3.weight, std=0.001)
        init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance4.weight, std=0.001)
        init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance5.weight, std=0.001)
        init.constant(self.instance5.bias, 0)

        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 24, 8]
        x = self.base(x)
        f_shape = x.size()

        y = x.unsqueeze(1)
        y = F.avg_pool3d(x, (16, 1, 1)).squeeze(1)
        sx = int(x.size(2) / 6)
        kx = int(x.size(2) - sx * 5)
        x = F.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))  # H4 W8
        # ========================================================================#

        out0 = x.view(x.size(0), -1)
        out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        x = self.drop(x)
        x = self.local_conv(x)
        out1 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        x = self.feat_bn2d(x)
        x = F.relu(x)  # relu for local_conv feature

        x = x.chunk(6, 2)
        x0 = x[0].contiguous().view(x[0].size(0), -1)
        x1 = x[1].contiguous().view(x[1].size(0), -1)
        x2 = x[2].contiguous().view(x[2].size(0), -1)
        x3 = x[3].contiguous().view(x[3].size(0), -1)
        x4 = x[4].contiguous().view(x[4].size(0), -1)
        x5 = x[5].contiguous().view(x[5].size(0), -1)

        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        c4 = self.instance4(x4)
        c5 = self.instance5(x5)
        return out0, (c0, c1, c2, c3, c4, c5)
