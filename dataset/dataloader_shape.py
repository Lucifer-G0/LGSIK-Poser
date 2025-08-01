import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from human_body_prior.body_model.body_model import BodyModel

# Different from the AvatarPoser paper, we use male/female model for each sequence
bm_fname_male = os.path.join("body_models", "smplh/{}/model.npz".format("male"))
dmpl_fname_male = os.path.join(
    "body_models", "dmpls/{}/model.npz".format("male")
)

bm_fname_female = os.path.join("body_models", 'smplh/{}/model.npz'.format('female'))
dmpl_fname_female = os.path.join("body_models", 'dmpls/{}/model.npz'.format('female'))

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

bm_male = BodyModel(
    bm_fname=bm_fname_male,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_fname_male,
)

bm_female = BodyModel(
    bm_fname=bm_fname_female,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_fname_female
)


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


def load_data(dataset_path, split, **kwargs):
    motion_list = get_path(dataset_path, split)
    input_motion_length = kwargs["input_motion_length"]
    motion_raw_data = []
    for data_i in motion_list:
        motion_raw_data.extend(torch.load(data_i, weights_only=False))

    valid_motion_data = []
    for data in motion_raw_data:
        seq_len = data["rotation_local_full_gt_list"].shape[0]
        if split == "train" and seq_len < input_motion_length:
            continue

        window_size = input_motion_length

        gt_betas = data["body_parms_list"]['betas'][[0]]
        gt_gender = data["gender"].item().upper()
        bm = bm_male if gt_gender == 'MALE' else bm_female
        output = bm(betas=gt_betas)

        vertices = output.v[0]  # shape (N, 3) 同一文件形状相同，只用一帧即可计算
        y_max = vertices[:, 1].max()
        y_min = vertices[:, 1].min()
        height = y_max - y_min

        gender_map = {'MALE': [1, 0], 'FEMALE': [0, 1]}
        gender_encoded = gender_map[gt_gender]
        gh = torch.tensor(gender_encoded + [height], dtype=torch.float32)

        joints = output.Jtr[0,:22]  # shape (22, 3)
        joints = joints - joints[[0]]  # 根部清零
        # 手部腿部相对于根和头的位移，头和根相对位移 3*9
        j1 = [1, 2, 16, 17]
        j2 = [4, 5, 18, 19]
        j3 = [7, 8, 20, 21]
        blimb = torch.norm(joints[j1] - joints[j2], dim=-1)
        slimb = torch.norm(joints[j2] - joints[j3], dim=-1)

        sparse_offsets = torch.cat((gh, blimb, slimb), dim=-1).repeat(seq_len, 1)

        if split == "train":
            window_step = input_motion_length // 2
        else:
            window_step = input_motion_length //4
        frames = data["rotation_local_full_gt_list"].shape[0]
        for start in range(0, frames, window_step):
            end = start + window_size
            if end < frames:
                input_feat = data["hmd_position_global_full_gt_list"][start:end]
                gt_local_pose = data["rotation_local_full_gt_list"][start:end]
                gt_global_pose = data["rotation_global_full_gt_list"][start:end]
                gt_positions = data["position_global_full_gt_world"][start:end]
                gt_betas = data["body_parms_list"]['betas'][1:][start:end]
                gt_sparse_offsets = sparse_offsets[start:end]
                valid_motion_data.append({
                    'input_feat': input_feat,
                    'gt_local_pose': gt_local_pose,
                    'gt_global_pose': gt_global_pose,
                    'gt_positions': gt_positions,
                    'gt_betas': gt_betas,
                    'gt_sparse_offsets': gt_sparse_offsets
                })

    return valid_motion_data

def load_testdata(dataset_path, split, **kwargs):
    motion_list = get_path(dataset_path, split)
    input_motion_length = kwargs["input_motion_length"]
    motion_raw_data = []
    for data_i in motion_list:
        motion_raw_data.extend(torch.load(data_i, weights_only=False))

    valid_motion_data = []
    for data in motion_raw_data:
        seq_len = data["rotation_local_full_gt_list"].shape[0]
        if split == "train" and seq_len < input_motion_length:
            continue

        gt_betas = data["body_parms_list"]['betas'][[0]]
        gt_gender = data["gender"].item().upper()
        bm = bm_male if gt_gender == 'MALE' else bm_female
        output = bm(betas=gt_betas)

        vertices = output.v[0]  # shape (N, 3) 同一文件形状相同，只用一帧即可计算
        y_max = vertices[:, 1].max()
        y_min = vertices[:, 1].min()
        height = y_max - y_min
        height = 0  # unfixed training bug, temp repalce

        gender_map = {'MALE': [1, 0], 'FEMALE': [0, 1]}
        gender_encoded = gender_map[gt_gender]
        gh = torch.tensor(gender_encoded + [height], dtype=torch.float32)

        joints = output.Jtr[0,:22]  # shape (22, 3)
        joints = joints - joints[[0]]  # 根部清零
        # 手部腿部相对于根和头的位移，头和根相对位移 3*9
        j1 = [1, 2, 16, 17]
        j2 = [4, 5, 18, 19]
        j3 = [7, 8, 20, 21]
        blimb = torch.norm(joints[j1] - joints[j2], dim=-1)
        slimb = torch.norm(joints[j2] - joints[j3], dim=-1)

        sparse_offsets = torch.cat((gh, blimb, slimb), dim=-1).repeat(seq_len, 1)

        input_feat = data["hmd_position_global_full_gt_list"]
        gt_local_pose = data["rotation_local_full_gt_list"]
        gt_global_pose = data["rotation_global_full_gt_list"]
        gt_positions = data["position_global_full_gt_world"]
        gt_betas = data["body_parms_list"]['betas'][1:]
        gt_sparse_offsets = sparse_offsets
        valid_motion_data.append({
           'input_feat': input_feat,
           'gt_local_pose': gt_local_pose,
           'gt_global_pose': gt_global_pose,
           'gt_positions': gt_positions,
           'gt_betas': gt_betas,
           'gt_sparse_offsets': gt_sparse_offsets,
            'filepath':data["filepath"]
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
        gt_sparse_offsets = motion_data['gt_sparse_offsets']

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
        gt_sparse_offsets = gt_sparse_offsets.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        return input_feat_res, gt_local_pose, gt_global_pose, gt_positions, gt_betas,gt_sparse_offsets


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
        gt_sparse_offsets = motion_data['gt_sparse_offsets']

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
        gt_sparse_offsets = gt_sparse_offsets.unsqueeze(0).repeat(input_feat_res.shape[0], 1, 1)
        return input_feat_res, gt_local_pose, gt_global_pose, gt_positions, gt_betas,gt_sparse_offsets


class EvalDataset(Dataset):
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

        input_feat = motion_data['input_feat']
        gt_local_pose = motion_data['gt_local_pose']
        gt_global_pose = motion_data['gt_global_pose']
        gt_positions = motion_data['gt_positions']
        gt_betas = motion_data['gt_betas']
        gt_sparse_offsets = motion_data['gt_sparse_offsets']
        filepath = motion_data['filepath']

        if 'HMD_3IMUs' in self.compatible_inputs:
            pass
        if 'HMD_2IMUs' in self.compatible_inputs:
            input_feat[:, list(range(30, 36)) + list(range(66, 72)) + list(range(132, 135))] = 0
        if 'HMD' in self.compatible_inputs:
            input_feat[:, list(range(18, 36)) + list(range(54, 72)) + list(range(126, 135))] = 0

        return input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas,gt_sparse_offsets,filepath
