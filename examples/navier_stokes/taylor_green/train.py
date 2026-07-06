import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.optim as optim

from src.dataset import create_training_dataset, create_test_dataset
from src.model import PINNModel, NavierStokes2D
from src.utils import calculate_l2_error, visual

# Set random seeds for reproducibility
torch.manual_seed(123456)
np.random.seed(123456)

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="navier stokes train (PyTorch)")
    parser.add_argument("--config_file_path", type=str, default="./configs/taylor_green_2D.yaml")
    return parser.parse_args()

def train(args):
    '''Train and evaluate the network'''
    config = load_yaml_config(args.config_file_path)
    
    # 自动探测设备：CUDA 优先
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Create datasets
    train_loader = create_training_dataset(config)
    eval_inputs, eval_label = create_test_dataset(config)

    # 缩放参数计算
    coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
    coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)
    input_center = list(0.5 * (coord_max + coord_min))
    input_scale = list(2.0 / (coord_max - coord_min))

    # 初始化模型
    model = PINNModel(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        layers=config["model"]["layers"],
        neurons=config["model"]["neurons"],
        input_scale=input_scale,
        input_center=input_center
    ).to(device)

    # 优化器与问题定义
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["optimizer"]["learning_rate"]
    )
    problem = NavierStokes2D(model, re=config["summary"]["Re"])

    epochs = config["data"]["train"]["epochs"]
    steps_per_epochs = len(train_loader)
    print(f"number of steps_per_epochs: {steps_per_epochs}")

    for epoch in range(1, 1 + epochs):
        model.train()
        local_time_beg = time.time()
        
        step_train_loss = 0.0
        for batch_pde, batch_ic, batch_bc in train_loader:
            batch_pde = batch_pde.to(device)
            batch_ic = batch_ic.to(device)
            batch_bc = batch_bc.to(device)

            optimizer.zero_grad()
            loss = problem.get_loss(batch_pde, batch_ic, batch_bc)
            loss.backward()
            optimizer.step()
            
            step_train_loss = loss.item()

        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds / steps_per_epochs) * 1000
        
        print(f"epoch: {epoch} train loss: {step_train_loss:.6f} "
              f"epoch time: {epoch_seconds:5.3f}s step time: {step_seconds:5.3f}ms")

        # 定期评估计算 L2 误差
        if epoch % config["summary"]["eval_interval_epochs"] == 0:
            calculate_l2_error(model, eval_inputs, eval_label, config, device)

    # 训练结束后进行可视化并生成视频/图像
    visual(model, epochs, eval_inputs, eval_label, device)

if __name__ == '__main__':
    print("pid:", os.getpid())
    args = parse_args()
    train(args)