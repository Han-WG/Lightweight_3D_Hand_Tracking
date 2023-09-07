# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
from common.nets.mobileNetv2_FPN import MobileNetV2

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        # self.resnet = ResNetBackbone(cfg.resnet_type).cuda()
        self.mobilenet = MobileNetV2()

    def forward(self, img):
        points = self.mobilenet.forward(img.cuda())

        return points
