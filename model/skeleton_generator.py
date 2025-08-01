import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Generator6(nn.Module):
    def __init__(self, device, parents=None, channels_per_joint=9, kernel_size=3):
        super(Generator6, self).__init__()
        if parents is None:
            parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]  #AMASS
        self.device = device
        self.layers = nn.ModuleList()
        self.parents = parents
        self.pooling_lists = []
        self.channel_list = []
        self.channels_per_joint = channels_per_joint

        number_layers = 3
        padding = (kernel_size - 1) // 2
        neighbor_list9 = [[0, 1, 2, 3, 4, 5, 7, 8], [0, 1, 6, 8], [0, 2, 6, 8], [0, 3, 6, 8], [0, 4, 7, 8],
                          [0, 5, 7, 8], [1, 2, 3, 6, 8],
                          [0, 4, 5, 7, 8],[8]]
        neighbor_list6 = [[0, 1, 2, 3, 4, 5, 7, 8], [0, 1, 6, 8], [0, 2, 6, 8], [0, 3, 6, 8], [0, 4, 7, 8],
                          [0, 5, 7, 8]]
        neighbor_list = [neighbor_list9, neighbor_list9, neighbor_list6]
        in_joint_num = [9, 9, 9]
        out_joint_num = [9, 9, 6]

        for i in range(number_layers):
            seq = []
            seq.append(
                SkeletonConv(
                    neighbor_list=neighbor_list[i],
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channels_per_joint,
                    out_channels_per_joint=self.channels_per_joint,
                    in_joint_num=in_joint_num[i],
                    out_joint_num=out_joint_num[i],
                    padding=padding,
                    stride=1,
                    device=device
                )
            )
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

    def save(self, epoch, filename):
        torch.save({'state_dict': self.state_dict(), 'epoch': epoch}, filename)

    def load(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])

    def forward(self, x_in):
        """
        骨架时空卷积运算
        :param x_in (batch_size,frames,joint_num,channels_per_joint)
        :return: A tuple containing:
         - generated_9d: (batch_size,frames, joint_num,channels_per_joint)
        """
        batch_size = x_in.shape[0]
        time_seq = x_in.shape[1]
        x = x_in.reshape(batch_size, time_seq, -1).permute(0, 2, 1)  # batch_size,joint_num*channels_per_joint,frames

        for i, layer in enumerate(self.layers):
            x = layer(x)

        # change input to shape (batch_size, frames, num_joints, channels_per_joint)
        result = x.permute(0, 2, 1).reshape(batch_size, time_seq, 6, -1)

        return result


class SkeletonConv(nn.Module):
    def __init__(
            self,
            neighbor_list,
            kernel_size,
            in_channels_per_joint,
            out_channels_per_joint,
            in_joint_num,
            out_joint_num,
            padding,
            stride,
            device,
    ):
        super().__init__()

        self.neighbor_list = neighbor_list
        self.device = device
        self.padding = padding
        self.stride = stride
        self._padding_repeated_twice = (padding, padding)
        self.padding_mode = "reflect"

        in_channels = in_channels_per_joint * in_joint_num
        out_channels = out_channels_per_joint * out_joint_num
        self.in_channels_per_joint = in_channels_per_joint
        self.out_channels_per_joint = out_channels_per_joint

        # expanded points use channels instead of joints
        self.expanded_neighbor_list = create_expanded_neighbor_list(
            neighbor_list, self.in_channels_per_joint
        )

        # weight is a matrix of size (out_channels, in_channels, kernel_size) containing the
        # convolution parameters learned from the data in a temporal window of kernel_size
        self.weight = torch.zeros(out_channels, in_channels, kernel_size).to(
            self.device
        )
        self.bias = torch.zeros(out_channels).to(self.device)
        # mask is a matrix of size (out_channels, in_channels, kernel_size) containing
        # which channels of the input affect the output (repeated in the dim=2)
        self.mask = torch.zeros_like(self.weight).to(self.device)

        self.description = (
            "SkeletonConv(in_channels_per_joint={}, out_channels_per_joint={}, kernel_size={}, "
            " stride={}, padding={})".format(
                in_channels_per_joint,
                out_channels_per_joint,
                kernel_size,
                stride,
                padding,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):

        for i, neighbor in enumerate(self.expanded_neighbor_list):
            """Use temporary variable to avoid assign to copy of slice, which might lead to un expected result"""
            tmp = torch.zeros_like(
                self.weight[
                self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                neighbor,
                ...,
                ]
            )
            for j in self.neighbor_list[i]:
                self.mask[
                self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                j * self.in_channels_per_joint: (j + 1) * self.in_channels_per_joint,
                ...,
                ] = 1  # 调整为正则化邻接矩阵

            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
            self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
            neighbor,
            ...,
            ] = tmp
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight[
                self.out_channels_per_joint
                * i: self.out_channels_per_joint
                     * (i + 1),
                neighbor,
                ...,
                ]
            )
            bound = 1 / math.sqrt(fan_in)
            tmp = torch.zeros_like(
                self.bias[
                self.out_channels_per_joint
                * i: self.out_channels_per_joint
                     * (i + 1)
                ]
            )
            nn.init.uniform_(tmp, -bound, bound)
            self.bias[
            self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)
            ] = tmp

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.bias = nn.Parameter(self.bias)

        # 冻结非邻接矩阵的权重
        with torch.no_grad():
            frozen_weights = (self.mask == 0)  # 定义冻结的掩码
            # 将这些部分的权重从计算图中分离，并禁止梯度计算
            self.weight[frozen_weights] = self.weight[frozen_weights].detach()  # 断开梯度连接

    def forward(self, input):
        # pytorch is channel first:
        # weights = (out_channels, in_channels, kernel1D_size)
        weight_masked = self.weight * self.mask
        res = F.conv1d(
            F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
            weight_masked,
            self.bias,
            self.stride,
            padding=0,
            dilation=1,
            groups=1
        )

        return res


def create_expanded_neighbor_list(neighbor_list, in_channels_per_joint):
    expanded_neighbor_list = []
    # create expanded_neighbor_list by appending channels of each neighbor joint
    for neighbor in neighbor_list:
        expanded = []
        for k in neighbor:
            for i in range(in_channels_per_joint):
                expanded.append(k * in_channels_per_joint + i)
        expanded_neighbor_list.append(expanded)
    return expanded_neighbor_list


def find_neighbor(parents, max_dist, add_displacement=False):
    distance_mat = calc_distance_mat(parents)
    neighbor_list = []  # for each joints contains all joints at max distance max_dist
    joint_num = len(parents)
    for i in range(joint_num):
        neighbor = []
        for j in range(joint_num):
            if distance_mat[i, j] <= max_dist:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    # displacement is treated as another joint (appended)
    if add_displacement:
        displacement = joint_num
        displacement_neighbor = neighbor_list[0].copy()
        for i in displacement_neighbor:
            neighbor_list[i].append(displacement)
        displacement_neighbor.append(displacement)  # append itself
        neighbor_list.append(displacement_neighbor)

    return neighbor_list


def calc_distance_mat(parents):
    """
    Parameters
    ----------
    parents : numpy.ndarray
        Parent indices of each joint.
    Returns
    -------
    distance_mat : numpy.ndarray
        Distance matrix len(parents) x len(parents) between any two joints
    """
    num_joints = len(parents)
    # distance_mat[i, j] = distance between joint i and joint j
    distance_mat = np.ones((num_joints, num_joints)) * np.inf
    for i in range(num_joints):
        distance_mat[i, i] = 0
    # calculate direct distances
    for i in range(num_joints):
        for j in range(num_joints):
            if i != j:
                d = distance_joints(parents, i, j)
                if d != 0:
                    distance_mat[i, j] = d
                    distance_mat[j, i] = d
    # calculate all other distances
    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                distance_mat[i][j] = min(
                    distance_mat[i][j], distance_mat[i][k] + distance_mat[k][j]
                )
    return distance_mat


def distance_joints(parents, i, j, init_dist=0):
    """
    Finds the distance between two joints if j is ancestor of i.
    Otherwise return 0.
    """
    if parents[i] == j:
        return init_dist + 1
    elif parents[i] == 0:
        return 0
    else:
        return distance_joints(parents, parents[i], j, init_dist + 1)
