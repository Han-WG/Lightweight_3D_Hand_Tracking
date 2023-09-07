# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
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


def main():
    print(torch.cuda.is_available())
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    # If you want to convert the model to onnx, modify common/nets/module.py p.18 :
    # from "points = self.mobilenet.forward(img.cuda())" to "points = self.mobilenet.forward(img)"
    with torch.no_grad():
        for itr, (inputs, targets) in enumerate(tqdm(tester.batch_generator)):
            # forward
            device = torch.device("cpu")
            model_ = torch.load("../output/model_dump/snapshot_0.pth", map_location=device)
            model = model_["model"]

            x = (inputs, targets, "test")
            export_onnx_file = "model_name.onnx"

            torch.onnx.export(model.module,  # pth file
                              x,
                              export_onnx_file,
                              opset_version=12,
                              do_constant_folding=True,
                              input_names=["input"],
                              output_names=["output"],
                              )
            break


if __name__ == "__main__":
    main()
