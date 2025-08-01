# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch


def pred_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pred_jitter = (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        ).norm(dim=2)



    return pred_jitter.mean(), pred_jitter.std(dim=0).mean()


def gt_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_jitter = (
        (
            (
                gt_position[3:]
                - 3 * gt_position[2:-1]
                + 3 * gt_position[1:-2]
                - gt_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
    )
    return gt_jitter.mean(), gt_jitter.std(dim=0).mean()


def mpjre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.absolute(diff)
    return rot_error.mean(), rot_error.std(dim=0).mean()


def mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error=torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
    mpjpe = torch.mean(pos_error)

    return mpjpe, pos_error.std(dim=0).mean()


def handpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_hands = (
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [20, 21]
        ]
    )
    return pos_error_hands.mean(),pos_error_hands.std(dim=0).mean()


def upperpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    upper_body_error = (
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., upper_index
        ]
    )
    return upper_body_error.mean(),upper_body_error.std(dim=0).mean()


def lowerpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    lower_body_error = (
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., lower_index
        ]
    )
    return lower_body_error.mean(),lower_body_error.std(dim=0).mean()


def rootpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_root = (
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [0]
        ]
    )
    return pos_error_root.mean(),pos_error_root.std(dim=0).mean()


def mpjve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    vel_error = (
        torch.sqrt(torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1))
    )
    return vel_error.mean(),vel_error.std(dim=0).mean()


metric_funcs_dict = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
