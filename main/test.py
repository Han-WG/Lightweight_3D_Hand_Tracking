# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import math

from tqdm import tqdm
from config import cfg
import params
import torch
from common.base import Tester
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', default='0', type=str, dest='test_epoch')
    parser.add_argument('--test_set', default="test", type=str, dest='test_set')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args


def main():
    global screen
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    point_loss = []
    params.main(tester)
    with torch.no_grad():
        for itr, (inputs, targets) in enumerate(tqdm(tester.batch_generator)):
            # forward
            out_x = tester.model(inputs, targets, 'test')
            targets = targets.reshape(-1, 63)
            x = torch.abs(out_x.cpu() - targets.cpu())
            relative_error = x / torch.abs(targets.cpu())
            loss_x = torch.mean(relative_error)
            loss_x = float(loss_x)
            if math.isinf(loss_x):
                continue
            point_loss.append(loss_x)

    # evaluate
    print(" Point_MRE : %.4f%% \n" % ((sum(point_loss) / len(point_loss)) * 100))
    screen = [" Point_MRE : %.4f%%" % ((sum(point_loss) / len(point_loss)) * 100)]
    tester.logger.info(' '.join(screen))
    return sum(point_loss) / len(point_loss)


if __name__ == "__main__":
    main()
