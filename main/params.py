# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from thop import profile, clever_format
from common.base import Trainer
import argparse
from config import cfg
import torch
from common.base import Tester
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', default='0', type=str, dest='test_epoch')
    parser.add_argument('--test_set', default="test", type=str, dest='test_set')
    args = parser.parse_args()

    return args


def main(trainer):
    print(torch.cuda.is_available())
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    # trainer = Trainer()
    # trainer._make_batch_generator()
    # trainer._make_model()

    with torch.no_grad():
        for itr, (inputs, targets) in enumerate(trainer.batch_generator):
            # forward
            model_ = torch.load("../output/model_dump/snapshot_0.pth")
            model_ = model_["model"].module

            flops, params = profile(model_, (inputs, targets, 'train'))
            # print(flops, params)
            macs, params = clever_format([flops, params], "%.3f")
            print('模型参数量： ', params, '计算量%.2f GFlops', macs)
            screen = ['模型参数量： ', params, '计算量: ', macs]
            trainer.logger.info(' '.join(screen))
            break


# if __name__ == "__main__":
#     main()
