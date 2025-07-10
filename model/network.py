import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from functools import partial

from model.skeleton_generator import Generator6
from model.sixdrotation import normalizeraw as normalize


class TemporalSpatialBackbone(torch.nn.Module):
    def __init__(self, device, hidden_size=512):
        super().__init__()
        self.device = device
        # 对原始输入按照节点数据进行拆分
        sparse_indexes = []  # 头、左手、右手、左腿、右腿、骨盆
        sparse_indexes.append(list(range(5 * 6, 6 * 6)) + list(range(36 + 5 * 6, 36 + 6 * 6)) +
                              list(range(126 + 2 * 3, 126 + 3 * 3))  # 加速度
                              )  # 骨盆
        sparse_indexes.append(
            list(range(0 * 6, 1 * 6)) + list(range(36 + 0 * 6, 36 + 1 * 6))
            + list(range(72 + 0 * 3, 72 + 1 * 3)) + list(range(81 + 0 * 3, 81 + 1 * 3))
        )  # 头
        sparse_indexes.append(
            list(range(1 * 6, 2 * 6)) + list(range(36 + 1 * 6, 36 + 2 * 6))
            + list(range(72 + 1 * 3, 72 + 2 * 3)) + list(range(81 + 1 * 3, 81 + 2 * 3))
        )  # 左手
        sparse_indexes.append(
            list(range(2 * 6, 3 * 6)) + list(range(36 + 2 * 6, 36 + 3 * 6))  # 全局旋转和角速度
            + list(range(72 + 2 * 3, 72 + 3 * 3)) + list(range(81 + 2 * 3, 81 + 3 * 3))  # 上半身位置和速度
        )  # 右手
        sparse_indexes.append(
            list(range(90 + 0 * 6, 90 + 1 * 6)) + list(range(102 + 0 * 6, 102 + 1 * 6))
            + list(range(114 + 0 * 3, 114 + 1 * 3)) + list(range(120 + 0 * 3, 120 + 1 * 3))
        )  # 左手 inhead
        sparse_indexes.append(
            list(range(90 + 1 * 6, 90 + 2 * 6)) + list(range(102 + 1 * 6, 102 + 2 * 6))  # 手相对头旋转和角速度
            + list(range(114 + 1 * 3, 114 + 2 * 3)) + list(range(120 + 1 * 3, 120 + 2 * 3))  # 手相对头位置和线速度
        )  # 右手 in head

        sparse_indexes.append(list(range(3 * 6, 4 * 6)) + list(range(36 + 3 * 6, 36 + 4 * 6)) +
                              list(range(126 + 0 * 3, 126 + 1 * 3))  # 加速度
                              )  # 左腿
        sparse_indexes.append(list(range(4 * 6, 5 * 6)) + list(range(36 + 4 * 6, 36 + 5 * 6)) +
                              list(range(126 + 1 * 3, 126 + 2 * 3))  # 加速度
                              )  # 右腿
        self.sparse_indexes = sparse_indexes

        pelvis_feats = sparse_indexes[0] + sparse_indexes[1] + sparse_indexes[4] + sparse_indexes[5]+ sparse_indexes[6] + sparse_indexes[7]
        head_feats =  sparse_indexes[1] + sparse_indexes[4] + sparse_indexes[5]
        lhand_feats = sparse_indexes[1] + sparse_indexes[2] + sparse_indexes[4]
        rhand_feats = sparse_indexes[1] + sparse_indexes[3] + sparse_indexes[5]
        lfoot_feats = sparse_indexes[0] + sparse_indexes[1] + sparse_indexes[6]
        rfoot_feats = sparse_indexes[0] + sparse_indexes[1] + sparse_indexes[7]
        upper_feats = sparse_indexes[0] + sparse_indexes[1]  + sparse_indexes[4] + sparse_indexes[5]
        lower_feats = sparse_indexes[0] + sparse_indexes[1] + sparse_indexes[6] + sparse_indexes[7]

        self.feat_indexes = [pelvis_feats, head_feats, lhand_feats, rhand_feats,
                             lfoot_feats, rfoot_feats, upper_feats, lower_feats]
        self.split_num = len(self.feat_indexes)
        split_hidden_size = hidden_size // self.split_num

        self.pelvis_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[0]), split_hidden_size),
                                                     nn.LeakyReLU())
        self.head_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[1]), split_hidden_size),
                                                   nn.LeakyReLU())
        self.lhand_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[2]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.rhand_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[3]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.lfoot_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[4]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.rfoot_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[5]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.upper_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[6]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.lower_linear_embedding = nn.Sequential(nn.Linear(len(self.feat_indexes[7]), split_hidden_size),
                                                    nn.LeakyReLU())
        self.shape_embedding = nn.Sequential(nn.Linear(11, split_hidden_size),
                                                    nn.LeakyReLU())

        self.embeddings = self.embeddings = [
            self.pelvis_linear_embedding,
            self.head_linear_embedding,
            self.lhand_linear_embedding,
            self.rhand_linear_embedding,
            self.lfoot_linear_embedding,
            self.rfoot_linear_embedding,
            self.upper_linear_embedding,
            self.lower_linear_embedding,
        ]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norms = nn.ModuleList([norm_layer(split_hidden_size) for _ in range(9)])

        num_rnn_layer = 1
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList(
                [torch.nn.LSTM(split_hidden_size, split_hidden_size, num_rnn_layer, bidirectional=False,
                               batch_first=True)
                 for _ in range(9)]),
                nn.ModuleList(
                    [torch.nn.LSTM(split_hidden_size, split_hidden_size, num_rnn_layer, bidirectional=False,
                                   batch_first=True)
                     for _ in range(6)])]
        )

        # iknet构造，结合原始输入和对应部分信息调整对应模块
        self.generator6 = Generator6(device, parents=None, channels_per_joint=split_hidden_size, kernel_size=1)

        out_joint_nums = [4, 2, 4, 4, 4, 4]  # 六个分区中除了头部都是四个节点
        self.spatial_encoder2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * split_hidden_size + len(self.sparse_indexes[0]+self.sparse_indexes[1]), 512), nn.LeakyReLU(),
                              nn.Linear(512, 256), nn.LeakyReLU(),
                              nn.Linear(256, out_joint_nums[0] * 6)
                              )]+[
                nn.Sequential(nn.Linear(4*6+split_hidden_size + len(self.sparse_indexes[i]), 512), nn.LeakyReLU(),
                              nn.Linear(512, 256), nn.LeakyReLU(),
                              nn.Linear(256, out_joint_nums[i] * 6)
                              )
                for i in range(1, len(out_joint_nums))]
        )

        # weight norm and initialization
        for encoder_i in self.time_encoder:
            for model_i in encoder_i:
                for layeridx in range(num_rnn_layer):
                    model_i = weightNorm(model_i, f"weight_ih_l{layeridx}")
                    model_i = weightNorm(model_i, f"weight_hh_l{layeridx}")
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)

    def forward(self, input_feat,sparse_offsets):
        """
        时空主干网络
        :param input_feat (batch_size,time_seq,135)
        :param sparse_offsets (batch_size,time_seq,11) 形状信息，性别，身高，大小臂长，大小腿长
        """
        batch_size, time_seq = input_feat.shape[0], input_feat.shape[1]
        # 使用列表中的嵌入层
        embeddings = []
        for i, embedding in enumerate(self.embeddings):
            x = self.norms[i](embedding(input_feat[..., self.feat_indexes[i]]))
            embeddings.append(x)
        embeddings.append(self.norms[8](self.shape_embedding(sparse_offsets)))

        collect_feats = torch.stack(embeddings, dim=-2).reshape(batch_size, time_seq, 9, -1)

        # block 0 LSTM Trans
        collect_feats_temporal = []
        for idx_num in range(9):
            collect_feats_temporal.append(self.time_encoder[0][idx_num](collect_feats[:, :, idx_num, :])[0])
        collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)

        collect_feats = self.generator6(collect_feats_temporal.reshape(batch_size, time_seq, -1))

        # block 2 IK
        shape_feats_temporal = []
        for idx_num in range(6):
            te2_result = self.time_encoder[1][idx_num](collect_feats[:, :, idx_num, :])[0]
            shape_feats_temporal.append(te2_result)
        shape_feats = torch.cat(shape_feats_temporal, dim=-1)

        pose_feats_temporal = []
        for idx_num in range(6):
            te2_result = shape_feats_temporal[idx_num]
            if idx_num == 0:
                se2_result = self.spatial_encoder2[idx_num](
                    torch.cat((input_feat[..., self.sparse_indexes[idx_num]+self.sparse_indexes[1]], te2_result,shape_feats_temporal[1]), dim=-1))
            else:
                se2_result = self.spatial_encoder2[idx_num](
                    torch.cat((input_feat[..., self.sparse_indexes[idx_num]], te2_result, pose_feats_temporal[0]),
                              dim=-1))
            pose_feats_temporal.append(se2_result)
        feats_spatial = torch.cat(pose_feats_temporal, dim=-1).reshape(batch_size, time_seq, 22, 6)

        pose_feats = normalize(feats_spatial)

        # collect-feat为各分区形式，需要重组为原始数据形式,训练数据和AMASS结构的要求
        # 0,3,6,9  12,15  13,16,18,20  14,17,19,21  1,4,7,10  2,5,8,11
        index_map = [0, 14, 18, 1, 15, 19, 2, 16, 20, 3, 17, 21, 4, 6, 10, 5, 7, 11, 8, 12, 9, 13]
        pred_pose = torch.index_select(pose_feats, -2, torch.tensor(index_map, device=self.device)).flatten(-2, -1)

        return pred_pose, shape_feats


class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, device, hidden_size=512):
        super().__init__()
        self.backbone = TemporalSpatialBackbone(device, hidden_size).to(device)

        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size // 8 * 6+11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, x_in,sparse_offsets):
        pred_pose, shape_feats = self.backbone(x_in,sparse_offsets)

        pred_shapes = self.shape_est(torch.cat((shape_feats,sparse_offsets),dim=-1))

        return pred_pose, pred_shapes
