from thop import profile
import torch
from tqdm import tqdm
import argparse
from utils.utils_config import load_config
from dataset.dataloader_shape import load_data, EvalDataset
from torch.utils.data import DataLoader
from model.hmd_imu_model import HMDIMUModel
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="options/test_config.yaml",
                        help="Path, where config file is stored")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configs = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_datas = load_data(configs.dataset_path, "test",input_motion_length=configs.input_motion_length)

    model = HMDIMUModel(configs, device)
    model.load(configs.resume_model)
    print(f"successfully resume checkpoint in {configs.resume_model}")

    model = model.eval().to(device)

    # 构造 dummy 输入
    input_feat = torch.randn(1, 40, 135).to(device)  # 你模型的主输入
    sparse_offsets = torch.randn(1, 40, 11).to(device)  # 你模型的主输入

    # 计算 FLOPs 和 参数
    flops, params = profile(model, inputs=(input_feat,sparse_offsets))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    all_input_times = []

    for input_type in configs.compatible_inputs:
        print(f"Eval on {input_type}")
        test_dataset = EvalDataset(test_datas, [input_type],
                                   configs.input_motion_length, 1)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=1,
            drop_last=False
        )

        input_times = []
        with torch.no_grad():
            for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas, gt_sparse_offsets) in tqdm(
                    enumerate(test_dataloader)):
                input_feat = input_feat.to(device).float()
                gt_sparse_offsets = gt_sparse_offsets.to(device).float()

                start_time = time.time()
                model(input_feat,gt_sparse_offsets)
                end_time = time.time()

                infer_time = end_time - start_time
                input_times.append(infer_time)
                all_input_times.append(infer_time)

        avg_time = sum(input_times) / len(input_times)
        print(f"[{input_type}] Average Inference Time per Sample: {avg_time * 1000:.2f} ms")

    # 所有类型平均推理时间
    total_avg_time = sum(all_input_times) / len(all_input_times)
    print(f"\n[ALL INPUT TYPES] Overall Average Inference Time per Sample: {total_avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
