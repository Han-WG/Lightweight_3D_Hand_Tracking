# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import numpy as np


class Config:
    # dataset
    dataset = 'InterHand2.6M'  # InterHand2.6M, RHD, STB

    # input, output
    input_img_shape = (224, 224)
    sigma = 2.5
    bbox_3d_size = 400  # depth axis
    bbox_3d_size_root = 400  # depth axis
    output_root_hm_shape = 64  # depth axis

    bbox_real = (200, 200)
    ## model
    resnet_type = 50  # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [35, 50]
    end_epoch = 100
    lr = 1e-3
    lr_dec_factor = 10
    train_batch_size = 32

    ## testing config
    test_batch_size = 32
    trans_test = 'rootnet'  # gt, rootnet

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'model_dump')
    result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 24
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    img_mean = 0
    datalist_len = 0
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

import random
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
