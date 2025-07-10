import torch
from tqdm import tqdm
import argparse
from utils.utils_config import load_config
from dataset.dataloader_shape import load_testdata, EvalDataset
from torch.utils.data import DataLoader, TensorDataset
from model.hmd_imu_model import HMDIMUModel
from utils import utils_transform
import math
from utils.metrics import get_metric_function
import prettytable as pt

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre", "mpjpe", "mpjve", "pred_jitter","handpe",
    "upperpe", "lowerpe", "rootpe"

]
gt_metrics = ["gt_jitter"]
all_metrics = pred_metrics + gt_metrics

metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "gt_jitter": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="options/test_config.yaml",
                        help="Path, where config file is stored")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_config(args.config)
    device = torch.device('cpu')
    test_datas = load_testdata(configs.dataset_path, "test", input_motion_length=configs.input_motion_length)

    model = HMDIMUModel(configs, device)
    model.load(configs.resume_model)
    print(f"Successfully resumed checkpoint from {configs.resume_model}")
    model.eval()

    tb = pt.PrettyTable()
    tb.field_names = ['Input_type'] + pred_metrics + gt_metrics

    all_logs = []

    for input_type in configs.compatible_inputs:
        print(f"Testing on {input_type}")
        test_dataset = EvalDataset(test_datas, [input_type], configs.input_motion_length, 1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        print("batch_num: ", len(test_dataloader))

        log = {}
        for metric in all_metrics:
            log[metric] = 0

        with torch.no_grad():
            for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas, gt_sparse_offsets) in tqdm(
                    enumerate(test_dataloader)
            ):
                input_feat = input_feat.to(device).float()
                gt_local_pose = gt_local_pose.to(device).float()
                gt_positions = gt_positions.to(device).float()
                gt_sparse_offsets = gt_sparse_offsets.to(device).float()

                frames = input_feat.shape[1]
                if frames < configs.input_motion_length:
                    print(frames)
                    continue
                window_input = []
                window_gso = []
                for start in range(0, frames - configs.input_motion_length + 1, 1):
                    end = start + configs.input_motion_length
                    if end <= frames:
                        window_input.append(input_feat[0,start:end])
                        window_gso.append(gt_sparse_offsets[0,start:end])

                input_tensor = torch.stack(window_input, dim=0)  # (N, 40, D)
                gso_tensor = torch.stack(window_gso, dim=0)  # (N, 40, D)

                pred_poses = []
                pred_positions = []
                test_dataset = TensorDataset(input_tensor,gso_tensor)
                input_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
                for (input_batch,gso_batch) in input_loader:  # (batch_size,window_size,D)
                    pred_local_pose, _, _, pred_joint_position = model(input_batch.to(device),gso_batch.to(device))  # input_batch是tuple
                    pred_poses.append(pred_local_pose[:, -1])  # 只取最后一帧 (batch_size,D)
                    pred_positions.append(pred_joint_position[:, -1])  # 只取最后一帧 (batch_size,D)

                pred_local_pose = torch.cat(pred_poses, dim=0).unsqueeze(dim=0)  # 1,frames-39,D
                pred_joint_position = torch.cat(pred_positions, dim=0).unsqueeze(dim=0)
                gt_local_pose = gt_local_pose[:, 39:]
                gt_positions = gt_positions[:, 39:]

                batch_size, time_seq = pred_local_pose.shape[0], pred_local_pose.shape[1]

                pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6)).reshape(
                    batch_size * time_seq, 22 * 3)
                gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6)).reshape(batch_size * time_seq,
                                                                                                 22 * 3)
                gt_positions = gt_positions.reshape(batch_size * time_seq, 22, 3)
                pred_joint_position = pred_joint_position.reshape(batch_size * time_seq, 22, 3)
                pred_joint_position = pred_joint_position - pred_joint_position[:, 15:16] + gt_positions[:, 15:16]

                predicted_angle = pred_local_pose_aa[..., 3:66]
                predicted_root_angle = pred_local_pose_aa[..., :3]
                predicted_position = pred_joint_position

                gt_angle = gt_local_pose_aa[..., 3:66]
                gt_root_angle = gt_local_pose_aa[..., :3]

                upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
                eval_log = {}
                for metric in all_metrics:
                    eval_log[metric] = (
                        get_metric_function(metric)(
                            predicted_position,
                            predicted_angle,
                            predicted_root_angle,
                            gt_positions,
                            gt_angle,
                            gt_root_angle,
                            upper_index,
                            lower_index,
                            fps=60,
                        )
                        .cpu()
                        .numpy()
                    )

                for key in eval_log:
                    log[key] += eval_log[key]

            all_logs.append(log)
            tb.add_row([input_type] +
                       ['%.2f' % (log[metric] / len(test_dataloader) * metrics_coeffs[metric]) for metric in
                        pred_metrics] +
                       ['%.2f' % (log[metric] / len(test_dataloader) * metrics_coeffs[metric]) for metric in gt_metrics]
                       )
    # 添加平均行
    avg_log = {metric: sum([log[metric] for log in all_logs])/ len(test_dataloader) / len(all_logs) for metric in all_metrics}
    tb.add_row(['Average'] +
               ['%.2f' % (avg_log[metric] * metrics_coeffs[metric]) for metric in pred_metrics] +
               ['%.2f' % (avg_log[metric] * metrics_coeffs[metric]) for metric in gt_metrics]
               )

    print(tb)


if __name__ == "__main__":
    main()
