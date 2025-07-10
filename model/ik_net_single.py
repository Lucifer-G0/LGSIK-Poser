import numpy as np
import torch
import torch.nn as nn

from model.sixdrotation import normalize

from human_body_prior.body_model.body_model import BodyModel
import os
from utils import utils_transform


class IK_NET(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # 对原始输入按照节点数据进行拆分
        sparse_indexes = []  # 头、左手、右手、左腿、右腿、骨盆
        sparse_indexes.append(
            list(range(0 * 6, 1 * 6)) + list(range(36 + 0 * 6, 36 + 1 * 6))
            + list(range(72 + 0 * 3, 72 + 1 * 3)) + list(range(81 + 0 * 3, 81 + 1 * 3))
        )  # 头
        sparse_indexes.append(
            list(range(1 * 6, 2 * 6)) + list(range(36 + 1 * 6, 36 + 2 * 6))
            + list(range(72 + 1 * 3, 72 + 2 * 3)) + list(range(81 + 1 * 3, 81 + 2 * 3))
            + list(range(90 + 0 * 6, 90 + 1 * 6)) + list(range(102 + 0 * 6, 102 + 1 * 6))
            + list(range(114 + 0 * 3, 114 + 1 * 3)) + list(range(120 + 0 * 3, 120 + 1 * 3))
        )  # 左手
        sparse_indexes.append(
            list(range(2 * 6, 3 * 6)) + list(range(36 + 2 * 6, 36 + 3 * 6))  # 全局旋转和角速度
            + list(range(72 + 2 * 3, 72 + 3 * 3)) + list(range(81 + 2 * 3, 81 + 3 * 3))  # 上半身位置和速度
            + list(range(90 + 1 * 6, 90 + 2 * 6)) + list(range(102 + 1 * 6, 102 + 2 * 6))  # 手相对头旋转和角速度
            + list(range(114 + 1 * 3, 114 + 2 * 3)) + list(range(120 + 1 * 3, 120 + 2 * 3))  # 手相对头位置和线速度

        )  # 右手

        sparse_indexes.append(list(range(3 * 6, 4 * 6)) + list(range(36 + 3 * 6, 36 + 4 * 6)) +
                              list(range(126 + 0 * 3, 126 + 1 * 3))  # 加速度
                              )  # 左腿
        sparse_indexes.append(list(range(4 * 6, 5 * 6)) + list(range(36 + 4 * 6, 36 + 5 * 6)) +
                              list(range(126 + 1 * 3, 126 + 2 * 3))  # 加速度
                              )  # 右腿
        sparse_indexes.append(list(range(5 * 6, 6 * 6)) + list(range(36 + 5 * 6, 36 + 6 * 6)) +
                              list(range(126 + 2 * 3, 126 + 3 * 3))  # 加速度
                              )  # 骨盆
        self.sparse_indexes = sparse_indexes

        # 默认情况下的标准静态偏移
        support_dir = "./body_models/"
        subject_gender = "neutral"
        bm_fname = os.path.join(support_dir, 'smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = os.path.join(support_dir, 'dmpls/{}/model.npz'.format(subject_gender))
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        self.bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)

        parents = self.bm.kintree_table[0][:22].long()
        self.parents = parents

        channels_per_joint = 9
        hidden_size = 256

        # offsets leg: LowerLeg, Foot, Toe  # offsets arm: UpperArm, LowerArm, Hand
        # dq leg: UpperLeg, LowerLeg, Foot  # dq arm: Shoulder, UpperArm, LowerArm

        # Left Leg ----------------------------------------
        l_leg_offsets = [4, 7, 10]
        l_leg_dq_in = [7]
        self.l_leg_dq_out = [1, 4, 7]
        self.l_leg_dq_out = extend_channels(self.l_leg_dq_out, channels_per_joint)
        # Right Leg ----------------------------------------
        r_leg_offsets = [5, 8, 11]
        r_leg_dq_in = [8]
        self.r_leg_dq_out = [2, 5, 8]
        self.r_leg_dq_out = extend_channels(self.r_leg_dq_out, channels_per_joint)
        # Left Arm ----------------------------------------
        l_arm_offsets = [16, 18, 20]
        l_arm_dq_in = [18]
        self.l_arm_dq_out = [13, 16, 18]
        self.l_arm_dq_out = extend_channels(self.l_arm_dq_out, channels_per_joint)
        # Right Arm ----------------------------------------
        r_arm_offsets = [17, 19, 21]
        r_arm_dq_in = [19]
        self.r_arm_dq_out = [14, 17, 19]
        self.r_arm_dq_out = extend_channels(self.r_arm_dq_out, channels_per_joint)

        self.mainNet = TrunkNet(sparse_indexes[0] + sparse_indexes[1] + sparse_indexes[2] + sparse_indexes[5],
                                channels_per_joint, hidden_size, parents, device)

        self.larmNet = LimbNet(channels_per_joint, hidden_size, parents,
                               sparse_indexes[1] + sparse_indexes[5], l_arm_offsets, l_arm_dq_in, device)
        self.rarmNet = LimbNet(channels_per_joint, hidden_size, parents,
                               sparse_indexes[2] + sparse_indexes[5], r_arm_offsets, r_arm_dq_in, device)
        self.llegNet = LimbNet(channels_per_joint, hidden_size, parents,
                               sparse_indexes[3] + sparse_indexes[5], l_leg_offsets, l_leg_dq_in, device)
        self.rlegNet = LimbNet(channels_per_joint, hidden_size, parents,
                               sparse_indexes[4] + sparse_indexes[5], r_leg_offsets, r_leg_dq_in, device)

    def save(self, epoch, filename):
        torch.save({'state_dict': self.state_dict(), 'epoch': epoch}, filename)

    def fk_module(self, global_orientation, joint_rotation, body_shape):
        """
            根据根的位置和局部节点旋转计算前22个关节的位置
        """
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1, 6)).reshape(
            global_orientation.shape[0], -1).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1, 6)).reshape(joint_rotation.shape[0],
                                                                                        -1).float()
        body_pose = self.bm(**{
            'root_orient': global_orientation,
            'pose_body': joint_rotation,
            'betas': body_shape.float()
        })
        joint_position = body_pose.Jtr[:, :22]
        return joint_position

    def load(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])

    def inverse_kinematics_R(self, R_global):
        """
        从全局旋转的6D表示法求出局部旋转的6D表示法。
        :param R_global: 全局旋转矩阵，形状为 [batch_size, num_joint, 3,3]
        :param parent: 父关节ID列表，形状为 [num_joint],父关节索引必须小于子关节
        :return: 局部旋转的6D表示法，形状为 [batch_size, num_joint, 3]
        """
        parent = self.parents

        # 初始化局部旋转矩阵
        R_local = torch.zeros_like(R_global)

        # 计算局部旋转矩阵
        for i in range(0, len(parent)):
            if parent[i] == -1:
                R_local[:, i] = R_global[:, i]
            else:
                R_local[:, i] = torch.bmm(torch.inverse(R_global[:, parent[i]]), R_global[:, i])

        return R_local

    # pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position = model(input_feat)
    def forward(self, pred_local_pose, rotation_global_r6d, pred_joint_position, input_feat, pred_shapes):
        """
        :return: A tuple containing:
         - pred_pose: 预测关节局部旋转(batch_size, frames, 22 * 6)
         - pred_global_pose: 预测关节全局旋转，6D表示法 (batch_size, frames, 22 * 6)
         - pred_joint_position: 预测关节位置（无根位移未对齐） (batch_size, frames, 22, 3)
        """
        # rotation_global_r6d shape: (batch_size, frames,22*6)
        # pred_joint_position shape: (batch_size, frames,22,3)
        batch_size = rotation_global_r6d.shape[0]
        frames = rotation_global_r6d.shape[1]

        rotations = rotation_global_r6d.reshape(batch_size, frames, 22, 6)
        positions = pred_joint_position
        motions = torch.cat((rotations, positions), dim=-1).flatten(-2, -1)  # (batch_size, frames,22,9)

        self.mainNet.run(input_feat, motions)
        self.larmNet.run(input_feat, motions, self.l_arm_dq_out)
        self.rarmNet.run(input_feat, motions, self.r_arm_dq_out)
        self.llegNet.run(input_feat, motions, self.r_leg_dq_out)
        self.rlegNet.run(input_feat, motions, self.r_leg_dq_out)

        pred_global_pose = motions.reshape(batch_size, frames, 22, 9)[:, :, :, :6].flatten(-2, -1)

        # 从全局旋转恢复出局部旋转
        rotation_global_matrot = utils_transform.sixd2matrot(pred_global_pose.reshape(-1, 6)).reshape(-1, 22, 3, 3)
        rotation_local_matrot = self.inverse_kinematics_R(rotation_global_matrot)
        pred_pose = utils_transform.matrot2sixd(rotation_local_matrot.reshape(-1, 3, 3)).reshape(batch_size, frames,
                                                                                                 22 * 6)

        pred_joint_position = self.fk_module(pred_pose[:, :, :6].reshape(-1, 6),
                                             pred_pose[:, :, 6:].reshape(-1, 21 * 6), pred_shapes.reshape(-1, 16))
        pred_joint_position = pred_joint_position.reshape(batch_size, frames, 22, 3)

        return pred_pose, pred_global_pose, pred_joint_position  # 暂不修改形状参数


class TrunkNet(nn.Module):
    def __init__(self, sparse_indexes, channels_per_joint, hidden_size, parents, device):
        super().__init__()

        dq_in_index = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        self.dq_in_extended = None
        self.channels_per_joint = channels_per_joint
        self.hidden_size = hidden_size
        self.device = device
        self.sparse_indexes = sparse_indexes

        self.dq_in_extended = extend_channels(dq_in_index, channels_per_joint, parents)

        self.sequential1 = self.create_sequential(1)
        self.sequential2 = self.create_sequential(3)
        self.sequential3 = self.create_sequential(3)

        dq_out_index1 = [9]
        dq_out_index2 = [6, 9, 12]
        dq_out_index3 = [0, 3, 6]
        self.dq_out_extended1 = extend_channels(dq_out_index1, self.channels_per_joint)
        self.dq_out_extended2 = extend_channels(dq_out_index2, self.channels_per_joint)
        self.dq_out_extended3 = extend_channels(dq_out_index3, self.channels_per_joint)

    def create_sequential(self, modify_node_num):
        input_size = (len(self.dq_in_extended)  # input pose
                      + len(self.sparse_indexes)  # sparse input (end effector)
                      )
        output_size = modify_node_num * self.channels_per_joint  # output modified joints

        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, output_size),
        ).to(self.device)

        return model

    def run(self, sparse_input, decoder_output):
        """
        :param sparse_input: 稀疏输入应包含根节点
        :param offsets:
        :param decoder_output:
        :return:
        """


        res1 = self.sequential1(
            torch.cat(
                (decoder_output[:, :, self.dq_in_extended],
                 sparse_input[:, :, self.sparse_indexes]), dim=2)
        )  # N,T,C

        decoder_output[:, :, self.dq_out_extended1] = res1

        res2 = self.sequential2(
            torch.cat(
                (decoder_output[:, :, self.dq_in_extended],
                 sparse_input[:, :, self.sparse_indexes]), dim=2)
        )

        decoder_output[:, :, self.dq_out_extended2] = res2

        res3 = self.sequential3(
            torch.cat((decoder_output[:, :, self.dq_in_extended], sparse_input[:, :, self.sparse_indexes]), dim=2))

        decoder_output[:, :, self.dq_out_extended3] = res3

        return res3


class LimbNet(nn.Module):
    def __init__(self, channels_per_joint, hidden_size,
                 parents, sparse_index, offsets_index, dq_in_index, device):
        super().__init__()
        self.channels_per_joint = channels_per_joint
        self.hidden_size = hidden_size
        self.device = device
        self.sparse_indexes = sparse_index

        self.dq_in_extended = extend_channels(dq_in_index, self.channels_per_joint, parents)

        self.sequential3 = self.create_sequential(3)

    def create_sequential(self, modify_node_num):
        input_size = (
                len(self.dq_in_extended)  # input pose
                + len(self.sparse_indexes)  # sparse input (end effector)
        )
        output_size = modify_node_num * self.channels_per_joint  # output modified joints

        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, output_size),
        ).to(self.device)

        return model

    def run(self, sparse_input, decoder_output, dq_out_extended):

        res3 = self.sequential3(
            torch.cat(
                (decoder_output[:, :, self.dq_in_extended],
                 sparse_input[:, :, self.sparse_indexes]), dim=2)
        )

        decoder_output[:, :, dq_out_extended] = res3

        return res3


def extend_channels(node_list, channels_per_joint, parents=None):
    # 将输入从 指定节点 拓展到 它到根节点的路径
    if parents is not None:
        while node_list[-1] != 0:
            node_list.append(parents[node_list[-1]])
    # 将原始列表拓展为通道列表
    channel_list = [
        i
        for j in node_list
        for i in range(j * channels_per_joint, j * channels_per_joint + channels_per_joint)
    ]
    return channel_list
