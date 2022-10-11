import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cayley_ortho_conv import Cayley, CayleyLinear
from block_ortho_conv import BCOP
from skew_ortho_conv import SOC

from custom_activations import *
from utils import conv_mapping, activation_mapping

class NormalizedLinear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight/weight_norm
        return F.linear(X, self.lln_weight if self.training else self.lln_weight.detach(), self.bias)

class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_layer, activation_name, stride=1, kernel_size=3):
        super(LipBlock, self).__init__()
        self.conv = conv_layer(in_planes, planes*stride, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.activation = activation_mapping(activation_name, planes*stride)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
        
class LipConvNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, lln=False):
        super(LipConvNet, self).__init__()        
        self.lln = lln
        self.in_planes = 3
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3)
        self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=1)
        
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        if self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)
        elif conv_name == 'cayley':
            self.last_layer = CayleyLinear(flat_features, num_classes)
        else:
            self.last_layer = conv_layer(flat_features, num_classes, 
                                         kernel_size=1, stride=1)

    def _make_layer(self, planes, num_blocks, conv_layer, activation, 
                    stride, kernel_size):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, 
                                   stride, kernel_size))
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x
        

class ResLayer(nn.Module):
    def __init__(self, in_planes, planes, num_blocks, conv_layer, activation, stride, kernel_size):
        super(ResLayer, self).__init__()
        self.num_blocks = num_blocks
        if self.num_blocks == 1:
            self.block = LipBlock(in_planes, planes, conv_layer, activation, stride, kernel_size)
        else:
            self.in_block = LipBlock(in_planes, planes, conv_layer, activation, stride=1, kernel_size=3)
            hidden_blocks = []
            for _ in range(num_blocks-2):
                cur_block = LipBlock(planes, planes, conv_layer, activation, stride=1, kernel_size=3)
                hidden_blocks.append(cur_block)
            self.hidden_block = nn.Sequential(*hidden_blocks)

            self.out_block = LipBlock(planes, planes, conv_layer, activation, stride, kernel_size)
        self.res_lamda_logit = nn.Parameter(torch.FloatTensor([0.0]))
        self.res_lamda_logit.requires_grad = False  # fixed lambda

    def forward(self, x):
        if self.num_blocks == 1:
            x = self.block(x)
        else:
            x = self.in_block(x)
            for block in self.hidden_block:
                ret = block(x)
                res_lamda = torch.sigmoid(self.res_lamda_logit)
                x = res_lamda*x+(1-res_lamda)*ret
            x = self.out_block(x)
        return x

class LipResNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, lln=False):
        super(LipResNet, self).__init__()        
        self.lln = lln
        self.in_planes = 3
        self.conv_name = conv_name
        self.input_side = input_side
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3)
        self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3)
        self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=1)
        
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        if self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)
        elif conv_name == 'cayley':
            self.last_layer = CayleyLinear(flat_features, num_classes)
        else:
            self.last_layer = conv_layer(flat_features, num_classes, 
                                         kernel_size=1, stride=1)

    def _make_layer(self, planes, num_blocks, conv_layer, activation, 
                    stride, kernel_size):
        layer = ResLayer(self.in_planes, planes, num_blocks, conv_layer, activation, stride, kernel_size)
        self.in_planes = planes * stride
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x

    def frozen_w_ortho(self,):
        assert self.conv_name == 'LOT'
        n = self.input_side
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            if layer.num_blocks == 1:
                layer.block.conv.frozen_w_ortho(n)
            else:
                layer.in_block.conv.frozen_w_ortho(n)
                for hidden in layer.hidden_block:
                    hidden.conv.frozen_w_ortho(n)
                layer.out_block.conv.frozen_w_ortho(n)
            n = n // 2

        if not self.lln:
            self.last_layer.frozen_w_ortho(n)
