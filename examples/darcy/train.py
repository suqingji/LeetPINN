"""train process in PyTorch"""
import argparse
import os
import time
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import create_training_dataset, create_test_dataset
from src.darcy import Darcy2D, FCSequential
from src.utils import visual, calculate_l2_error

torch.manual_seed(123456)
np.random.seed(123456)

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_log(*args):
    print(*args)

def parse_args():
    parser = argparse.ArgumentParser(description="darcy flow (PyTorch)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device: 'cuda' or 'cpu'")
    parser.add_argument("--config_file_path", type=str, default="./configs/darcy.yaml")
    return parser.parse_args()

def train(input_args):
    config = load_yaml_config(input_args.config_file_path)
    device = torch.device(input_args.device)
    print_log(f"Running on {device.type.upper()}.")

    # create train dataset
    train_dataset = create_training_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['train']["batch_size"], shuffle=True, drop_last=True)
    
    # create test dataset
    test_input, test_label = create_test_dataset(config)

    # network model
    model = FCSequential(in_channels=config["model"]["in_channels"],
                         out_channels=config["model"]["out_channels"],
                         neurons=config["model"]["neurons"],
                         layers=config["model"]["layers"],
                         residual=config["model"]["residual"],
                         act=config["model"]["activation"],
                         weight_init=config["model"]["weight_init"]).to(device)

    problem = Darcy2D(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])

    epochs = config["data"]["train"]["epochs"]
    steps_per_epochs = len(train_loader)
    print_log(f"number of steps_per_epochs: {steps_per_epochs}")

    for epoch in range(1, 1 + epochs):
        local_time_beg = time.time()
        model.train()
        cur_loss = 0.0
        
        for pde_data, bc_data in train_loader:
            pde_data = pde_data.to(device)
            bc_data = bc_data.to(device)
            
            optimizer.zero_grad()
            loss = problem.get_loss(pde_data, bc_data)
            loss.backward()
            optimizer.step()
            
            cur_loss = loss.item()

        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds / steps_per_epochs) * 1000
        print_log(f"epoch: {epoch} train loss: {cur_loss:.7f} "
                  f"epoch time: {epoch_seconds:5.3f}s step time: {step_seconds:5.3f}ms")

        if epoch % config["summary"]["eval_interval_epochs"] == 0:
            calculate_l2_error(model, test_input, test_label, config["data"]["train"]["batch_size"], device)

    visual(model, config, device)

if __name__ == "__main__":
    print("pid:", os.getpid())
    args = parse_args()
    train(args)