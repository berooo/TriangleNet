import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

import Constants

nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class TriangleNet_NoContour(nn.Module):
    def __init__(self, num_classes=Constants.BINARY_CLASS, num_channels=3):
        super(TriangleNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.content_decoder4 = DecoderBlock(filters[3], filters[2])
        self.content_decoder3 = DecoderBlock(filters[2], filters[1])
        self.content_decoder2 = DecoderBlock(filters[1], filters[0])
        self.content_decoder1 = DecoderBlock(filters[0], filters[0])

        self.content_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.content_finalrelu1 = nonlinearity
        self.content_finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.content_finalrelu2 = nonlinearity
        self.content_finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        ip = self.firstrelu(x)
        x = self.firstmaxpool(ip)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Content Decoder
        content_d4 = self.content_decoder4(e4) + e3
        content_d3 = self.content_decoder3(content_d4) + e2
        content_d2 = self.content_decoder2(content_d3) + e1
        content_d1 = self.content_decoder1(content_d2) + ip

        content_out = self.content_finaldeconv1(content_d1)
        content_out = self.content_finalrelu1(content_out)
        content_out = self.content_finalconv2(content_out)
        content_out = self.content_finalrelu2(content_out)
        content_out = self.content_finalconv3(content_out)

        return F.sigmoid(content_out)


