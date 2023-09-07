# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import math
def wing_loss(landmarks, labels, w=10., epsilon=2.):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    c = w * (1.0 - math.log(1.0 + w / epsilon))

    pre_points = landmarks.reshape(-1, 21, 3)
    gt_points = labels.reshape(-1, 21, 3)

    pre_cov = []
    gt_cov = []
    for i in pre_points:
        pre_cov.append(torch.cov(i))
    for i in gt_points:
        gt_cov.append(torch.cov(i))

    pre_cov = torch.stack(pre_cov).cuda()
    gt_cov = torch.stack(gt_cov).cuda()
    cov_loss = torch.abs(pre_cov - gt_cov)
    cov_losses = torch.where(
        (w > cov_loss),
        w * torch.log(1.0 + cov_loss / epsilon),
        cov_loss - c)
    cov_losses = torch.mean(cov_losses, dim=1, keepdim=True)
    cov_losses = torch.mean(cov_losses, dim=2, keepdim=True)

    pre_center = torch.mean(pre_points, dim=1).reshape(-1, 1, 3)
    gt_center = torch.mean(gt_points, dim=1).reshape(-1, 1, 3)
    pre_points = torch.cat((pre_points, pre_center), dim=1)
    gt_points = torch.cat((gt_points, gt_center), dim=1)

    x = pre_points - gt_points
    absolute_x = torch.abs(x)

    losses = torch.where(
        (w > absolute_x),
        w * torch.log(1.0 + absolute_x / epsilon),
        absolute_x - c)

    losses = torch.mean(losses, dim=1, keepdim=True)

    loss = torch.mean(0.6 * losses + 0.4 * cov_losses)
    return loss


class got_total_wing_loss(nn.Module):
    def __ini__(self):
        super(got_total_wing_loss, self).__init__()

    def forward(self, joint_out, joint_gt):
        loss = wing_loss(joint_out, joint_gt)
        return loss

