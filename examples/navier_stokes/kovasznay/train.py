import argparse
import os
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import create_dataset
from src.model import FCSequential, Kovasznay
from src.utils import calculate_l2_error, visual

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="kovasznay flow train (PyTorch)")
    parser.add_argument("--config_file_path", type=str, default="./configs/kovasznay_cfg.yaml",
                        help="config file path")
    return parser.parse_args()

def train(file_cfg):
    """Train and evaluate the network"""
    config = load_yaml_config(file_cfg)

    # 自动探测设备：CUDA 优先
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # create dataset
    ds_train = create_dataset(config)

    # create network
    model = FCSequential(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        layers=config["model"]["layers"],
        neurons=config["model"]["neurons"],
        residual=config["model"]["residual"],
    ).to(device)

    if config.get("load_ckpt", False):
        model.load_state_dict(torch.load(config["load_ckpt_path"]))

    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["initial_lr"])

    # create the problem
    problem = Kovasznay(model)

    def train_epoch(model, dataset, i_epoch):
        model.train()
        n_step = len(dataset)
        for i_step, (pde_data, bc_data) in enumerate(dataset):
            local_time_beg = time.time()
            
            pde_data = pde_data.to(device)
            bc_data = bc_data.to(device)

            optimizer.zero_grad()
            loss = problem.get_loss(pde_data, bc_data)
            loss.backward()
            optimizer.step()

            if i_step % 50 == 0:
                print(
                    "\repoch: {}, loss: {:>f}, time elapsed: {:.1f}ms [{}/{}]".format(
                        i_epoch,
                        float(loss.item()),
                        (time.time() - local_time_beg) * 1000,
                        i_step + 1,
                        n_step,
                    )
                )

    time_beg = time.time()
    for i_epoch in range(config["epochs"]):
        train_epoch(model, ds_train, i_epoch)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
    
    if config.get("save_ckpt", False):
        os.makedirs(os.path.dirname(config["save_ckpt_path"]), exist_ok=True)
        torch.save(model.state_dict(), config["save_ckpt_path"])

    # 绘制可视化图
    visual(model, config, device, resolution=config["visual_resolution"])

    # 测试相对 L2 误差
    n_samps = 10000 
    ds_test = create_dataset(config, n_samps)
    calculate_l2_error(problem, model, ds_test, device)


if __name__ == "__main__":
    print("pid:", os.getpid())
    args = parse_args()
    train(args.config_file_path)