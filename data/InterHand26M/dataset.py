# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data

import os.path as osp
from main.config import cfg
from common.utils.preprocessing import load_img, load_skeleton, process_bbox, augmentation, \
    transform_input_to_output_space, img_cut
from common.utils.transforms import world2cam, cam2pixel
import os
import json

from pycocotools.coco import COCO


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.mode = mode  # train, test, val
        self.annot_subset = mode
        self.img_path = '/home/hanwg/project/datasets/InterHand2.6m/images'
        self.annot_path = '/home/hanwg/project/datasets/InterHand2.6m/annotations'
        if self.mode == 'val':
            self.rootnet_output_path = '../data/InterHand26M/rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = '../data/InterHand26M/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)

        self.img_mean_list = 0
        self.img_mean = 0
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        for aid in db.anns.keys():
            if aid == 1:
                break
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_21 = joint_world[:21]
            joint_41 = joint_world[21:]
            if (np.std(joint_21[:, 0]) == 0.0) and joint_21[0] in joint_41:
                continue
            elif (np.std(joint_41[:, 0]) == 0.0) and joint_41[0] in joint_21:
                continue
            if (abs(max(list(joint_21[:, 0]))) > abs(min(list(joint_21[:, 0]))) * 100) | \
                    (abs(max(list(joint_41[:, 0]))) > abs(min(list(joint_41[:, 0]))) * 100):
                continue
            joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type'].lower()
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0],
                             'left': rootnet_result[str(aid)]['abs_depth'][1]}
                abs_depth_root = np.array(
                    (joint_cam[self.root_joint_idx['right'], 2], joint_cam[self.root_joint_idx['left'], 2]),
                    dtype=np.float32)
                root_valid = np.array((joint_valid[self.root_joint_idx['right']],
                                       joint_valid[self.root_joint_idx['left']]), dtype=np.float32)
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                             'left': joint_cam[self.root_joint_idx['left'], 2]}
                abs_depth_root = np.array(
                    (joint_cam[self.root_joint_idx['right'], 2], joint_cam[self.root_joint_idx['left'], 2]),
                    dtype=np.float32)
                root_valid = np.array((joint_valid[self.root_joint_idx['right']],
                                       joint_valid[self.root_joint_idx['left']]), dtype=np.float32)

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint,
                    'hand_type': hand_type, 'hand_type_valid': 0, 'abs_depth': abs_depth,
                    'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                    'annot_id': aid, 'root_valid': root_valid, 'abs_depth_root': abs_depth_root}

            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            # else:
            #     self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid, cam_param, abs_depth, root_valid_root_net, abs_depth_root \
            = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid'], \
            data['cam_param'], data['abs_depth'], data['root_valid'], data['abs_depth_root']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
        # image load
        img = load_img(img_path)

        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid,
                                                                           hand_type, self.mode, self.joint_type)

        coord = joint_coord.copy()
        rel_root_depth = np.array(
            [joint_coord[self.root_joint_idx['left'], 2] - joint_coord[self.root_joint_idx['right'], 2]],
            dtype=np.float32).reshape(1)
        root_valid = np.array(
            [joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],
            dtype=np.float32).reshape(1) if hand_type[0] * hand_type[1] == 1 else np.zeros((1), dtype=np.float32)
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord,
                                                                                               joint_valid,
                                                                                               rel_root_depth,
                                                                                               root_valid,
                                                                                               self.root_joint_idx,
                                                                                               self.joint_type)
        coord[:, 2] = joint_coord[:, 2] + cfg.input_img_shape[0] / 2
        img = self.transform(img.astype(np.float32)) / 255.

        if (int(hand_type[0]) == 1) & (int(hand_type[1]) == 0):
            targets = coord[:21, :]
        else:
            targets = coord[21:, :]

        targets = torch.Tensor(targets / cfg.input_img_shape[0])
        target_joint = targets.reshape(-1, 63)
        inputs = img
        return inputs, target_joint
