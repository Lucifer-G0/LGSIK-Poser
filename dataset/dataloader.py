import glob
import os

import torch
from torch.utils.data import Dataset


def get_path(dataset_path, split):
    data_list_path = []
    parent_data_path = glob.glob(dataset_path + "/*")
    for d in parent_data_path:
        if os.path.isdir(d):
            if os.path.exists(d + "/" + split):
                files = glob.glob(d + "/" + split + "/*pt")
                if len(files) > 0:
                    data_list_path.extend(files)
    return data_list_path


def load_data(dataset_path, split, input_motion_length):
    motion_list = get_path(dataset_path, split)
    motion_raw_data = []
    for data_i in motion_list:
        motion_raw_data.extend(torch.load(data_i, weights_only=False))

    valid_motion_data = []
    for data in motion_raw_data:
        if split == "train" and data["rotation_local_full_gt_list"].shape[0] < input_motion_length:
            continue

        window_size = input_motion_length
        if split == "train":
            window_step = input_motion_length // 2
        else:
            window_step = input_motion_length // 4
        frames = data["rotation_local_full_gt_list"].shape[0]
        for start in range(0, frames, window_step):
            end = start + window_size
            if end < frames:
                input_feat = data["hmd_position_global_full_gt_list"][start:end]
                gt_local_pose = data["rotation_local_full_gt_list"][start:end]
                gt_global_pose = data["rotation_global_full_gt_list"][start:end]
                gt_positions = data["position_global_full_gt_world"][start:end]
                gt_betas = data["body_parms_list"]['betas'][1:][start:end]
                valid_motion_data.append({
                    'input_feat': input_feat,
                    'gt_local_pose': gt_local_pose,
                    'gt_global_pose': gt_global_pose,
                    'gt_positions': gt_positions,
                    'gt_betas': gt_betas
                })

    return valid_motion_data


class TrainDataset(Dataset):
    def __init__(
            self,
            train_datas,
            compatible_inputs=['HMD', 'HMD_2IMUs', 'HMD_3IMUs'],
            input_motion_length=40,
            train_dataset_repeat_times=1,
    ):
        self.compatible_inputs = compatible_inputs
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length
        self.motions = train_datas

    def __len__(self):
        return len(self.motions) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        motion_data = self.motions[idx % len(self.motions)]

        gt_local_pose = motion_data['gt_local_pose']
        gt_global_pose = motion_data['gt_global_pose']
        gt_positions = motion_data['gt_positions']
        gt_betas = motion_data['gt_betas']

        input_feat_all = []
        if 'HMD_3IMUs' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat_all.append(input_feat)
        if 'HMD_2IMUs' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat[:, list(range(30, 36)) + list(range(66, 72)) + list(range(132, 135))] = 0
            input_feat_all.append(input_feat)
        if 'HMD' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat[:, list(range(18, 36)) + list(range(54, 72)) + list(range(126, 135))] = 0
            input_feat_all.append(input_feat)

        input_feat_res = torch.stack(input_feat_all, dim=0)
        gt_local_pose = gt_local_pose.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        gt_global_pose = gt_global_pose.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        gt_positions = gt_positions.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1, 1)
        gt_betas = gt_betas.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        return input_feat_res, gt_local_pose, gt_global_pose, gt_positions, gt_betas


class TestDataset(Dataset):
    def __init__(
            self,
            train_datas,
            compatible_inputs=['HMD', 'HMD_2IMUs', 'HMD_3IMUs'],
            input_motion_length=40,
            train_dataset_repeat_times=1,
    ):
        self.compatible_inputs = compatible_inputs
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length
        self.motions = train_datas

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion_data = self.motions[idx % len(self.motions)]

        gt_local_pose = motion_data['gt_local_pose']
        gt_global_pose = motion_data['gt_global_pose']
        gt_positions = motion_data['gt_positions']
        gt_betas = motion_data['gt_betas']

        input_feat_all = []
        if 'HMD_3IMUs' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat_all.append(input_feat)
        if 'HMD_2IMUs' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat[:, list(range(30, 36)) + list(range(66, 72)) + list(range(132, 135))] = 0
            input_feat_all.append(input_feat)
        if 'HMD' in self.compatible_inputs:
            input_feat = motion_data['input_feat'].clone()
            input_feat[:, list(range(18, 36)) + list(range(54, 72)) + list(range(126, 135))] = 0
            input_feat_all.append(input_feat)

        input_feat_res = torch.stack(input_feat_all, dim=0)
        gt_local_pose = gt_local_pose.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        gt_global_pose = gt_global_pose.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        gt_positions = gt_positions.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1, 1)
        gt_betas = gt_betas.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        return input_feat_res, gt_local_pose, gt_global_pose, gt_positions, gt_betas
