# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

from config import cfg
from config import set_seed
from common.base import Trainer
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, dest='gpu_ids')
    parser.add_argument('--continue', default='0', dest='continue_train', action='store_true')
    parser.add_argument('--seed', type=int, default=126673, help='seed')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    # argument parse and create log
    args = parse_args()
    set_seed(args.seed)
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    point_loss_cal = []
    epoch_num = []
    opt_loss = 1e7

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        index = 0
        joint_loss_epoch = 0
        for itr, (inputs, targets) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, 'train')
            loss = {k: loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            ]
            joint_loss_epoch += loss['joint'].detach().cpu()
            screen += ['%s: %.4f %s: %.4f' % ('loss_joint', loss['joint'].detach(),
                                              'loss_joint_sum', joint_loss_epoch
                                              )]
            if index % 1000 == 0:
                trainer.logger.info(' '.join(screen))
            index += 1
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        if joint_loss_epoch / trainer.itr_per_epoch < opt_loss:
            # save best model
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'model': trainer.model,
                'optimizer': trainer.optimizer.state_dict(),
            }, 0)
            opt_loss = joint_loss_epoch / trainer.itr_per_epoch
        if epoch == cfg.end_epoch - 1:
            # save finally model
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'model': trainer.model,
                'optimizer': trainer.optimizer.state_dict(),
            }, 1)

        # draw the detection result
        point_loss_cal.append(joint_loss_epoch / trainer.itr_per_epoch)
        epoch_num.append(epoch)
        trainer.logger.info("point_average_loss : %.4f" % (joint_loss_epoch / trainer.itr_per_epoch))
        plt.figure(facecolor='gray')
        plt.xlabel('epoch')
        plt.plot(epoch_num, point_loss_cal, label=u'point_loss')  # draw
        plt.legend(loc='upper right')
        plt.savefig('../output/model_dump/train_result.png')
        plt.close()
    params.main(trainer)


if __name__ == "__main__":
    main()
