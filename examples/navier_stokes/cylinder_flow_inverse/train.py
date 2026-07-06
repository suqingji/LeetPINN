import argparse
import os
import time
import yaml
import numpy as np

import torch
import torch.optim as optim

from src.dataset import create_training_dataset, create_test_dataset
from src.inv_navier_stokes import PINNModel, InvNavierStokes
from src.utils import calculate_l2_error, visual, plot_params

# 保证复现性
torch.manual_seed(0)
np.random.seed(0)

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="cylinder flow train (PyTorch)")
    parser.add_argument("--config_file_path", type=str, default="./configs/navier_stokes_inverse.yaml")
    return parser.parse_args()

def train(args):
    config = load_yaml_config(args.config_file_path)
    
    # 自动设置设备，优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 数据集创建
    train_loader = create_training_dataset(config)
    eval_inputs, eval_label = create_test_dataset(config)

    # 归一化参数计算
    coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
    coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)
    input_center = list(0.5 * (coord_max + coord_min))
    input_scale = list(2.0 / (coord_max - coord_min))

    # 初始化模型与参数
    model = PINNModel(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        layers=config["model"]["layers"],
        neurons=config["model"]["neurons"],
        input_scale=input_scale,
        input_center=input_center
    ).to(device)

    # 初始化反演参数 theta1 和 theta2
    theta = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32, device=device))
    
    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()) + [theta], 
        lr=config["optimizer"]["learning_rate"]
    )
    
    # PDE 损失构造器
    problem = InvNavierStokes(model, theta)

    epochs = config["data"]["train"]["epochs"]
    steps_per_epochs = len(train_loader)
    print(f"number of steps_per_epochs: {steps_per_epochs}")

    param1_hist = []
    param2_hist = []

    for epoch in range(1, 1 + epochs):
        model.train()
        local_time_beg = time.time()
        
        step_train_loss = 0.0
        for batch_points, batch_labels in train_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = problem.get_loss(batch_points, batch_labels)
            loss.backward()
            optimizer.step()
            
            step_train_loss = loss.item()

        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds / steps_per_epochs
        
        print(f"epoch: {epoch} train loss: {step_train_loss:.6f} "
              f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")

        # 定期评估与记录参数
        if epoch % config["summary"]["eval_interval_epochs"] == 0:
            current_theta = theta.detach().cpu().numpy()
            print(f"Params are {current_theta}")
            param1_hist.append(current_theta[0])
            param2_hist.append(current_theta[1])
            
            calculate_l2_error(model, eval_inputs, eval_label, config, device)

    # 训练结束后进行可视化
    visual(model, epochs, eval_inputs, eval_label, device)
    plot_params(param1_hist, param2_hist)

if __name__ == '__main__':
    print("pid:", os.getpid())
    args = parse_args()
    train(args)