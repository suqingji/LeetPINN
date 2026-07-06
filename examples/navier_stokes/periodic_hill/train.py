import argparse
import os
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import create_train_dataset, create_test_dataset
from src.model import FCSequential, NavierStokesRANS
from src.utils import calculate_l2_error, predict

# 设定随机种子以保证结果复现性
torch.manual_seed(0)
np.random.seed(0)

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="periodic hill train (PyTorch)")
    parser.add_argument("--config_file_path", type=str, default="./configs/rans.yaml")
    return parser.parse_args()

def train(input_args):
    '''Train and evaluate the network'''
    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optim_params = config["optimizer"]
    summary_params = config["summary"]

    # 自动设置设备，优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 创建训练集与测试集
    dataset_loader = create_train_dataset(data_params["data_path"], data_params["batch_size"])
    inputs, label = create_test_dataset(data_params["data_path"])

    # 实例化模型
    model = FCSequential(in_channels=model_params["in_channels"],
                         out_channels=model_params["out_channels"],
                         layers=model_params["layers"],
                         neurons=model_params["neurons"],
                         residual=model_params["residual"],
                         act=model_params["activation"]).to(device)

    # 检查点加载与保存目录初始化
    if summary_params.get("load_ckpt", False):
        model.load_state_dict(torch.load(summary_params["load_ckpt_path"]))
        print(f"Loaded checkpoint from {summary_params['load_ckpt_path']}")
        
    if not os.path.exists(os.path.abspath(summary_params['ckpt_path'])):
        os.makedirs(os.path.abspath(summary_params['ckpt_path']))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=optim_params["initial_lr"], weight_decay=optim_params["weight_decay"])
    
    # 问题实例化 (方程参数设置)
    problem = NavierStokesRANS(model, re=5600.0, rho=1.0)

    epochs = optim_params["train_epochs"]

    for epoch in range(1, 1 + epochs):
        time_beg = time.time()
        model.train()
        step_train_loss = 0.0
        
        # 训练迭代
        for pde_data, bc_coord, bc_labels in dataset_loader:
            pde_data = pde_data.to(device)
            bc_coord = bc_coord.to(device)
            bc_labels = bc_labels.to(device)
            
            optimizer.zero_grad()
            loss = problem.get_loss(pde_data, bc_coord, bc_labels)
            loss.backward()
            optimizer.step()
            
            step_train_loss = loss.item()
            
        print(f"epoch: {epoch} train loss: {step_train_loss:.6f} epoch time: {(time.time() - time_beg)*1000 :.3f}ms")
        
        # 定期评估与绘图
        if epoch % summary_params["eval_interval_epochs"] == 0:
            calculate_l2_error(model, inputs, label, config, device)
            predict(model=model, epochs=epoch, input_data=inputs, label=label, device=device, path=summary_params["visual_dir"])
            
        # 定期保存权重
        if epoch % summary_params["save_checkpoint_epochs"] == 0:
            ckpt_name = "rans_{}.ckpt".format(epoch)
            torch.save(model.state_dict(), os.path.join(summary_params['ckpt_path'], ckpt_name))

if __name__ == '__main__':
    print("pid:", os.getpid())
    args = parse_args()
    start_time = time.time()
    train(args)
    print(f"End-to-End total time: {time.time() - start_time} s")