# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
from common.nets.module import BackboneNet
from common.nets.loss import got_total_wing_loss
from config import cfg


class Model(nn.Module):
    def __init__(self, backbone_net):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net

        self.joint_heatmap_loss = got_total_wing_loss()
        self.loss = 0
        self.index = 0

    def render_keypoint(self, joint_coord):
        out = []
        for i in joint_coord:
            out.append(i / cfg.input_img_shape[0])
        out = torch.stack(out).cuda()
        return out

    def forward(self, inputs, targets, mode):
        targets = targets.reshape(-1, 63)
        input_img = inputs
        joint_heatmap_out = self.backbone_net(input_img)

        if mode == 'train':
            loss = {'joint': self.joint_heatmap_loss(joint_heatmap_out.cuda(), targets.cuda())
                    }

            return loss
        elif mode == 'test':

            return joint_heatmap_out


def get_model():
    backbone_net = BackboneNet()

    model = Model(backbone_net).cuda()
    return model
