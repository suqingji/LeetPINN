"""train process in PyTorch"""
import argparse
import os
import time
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# 这里我们只需要引入基础的库即可，去除MindFlow依赖
from src.dataset import create_training_dataset, create_test_dataset, BurgersDataset
from src.model import MLP_with_Residual, Burgers1D
from src.utils import print_log, visual, calculate_l2_error

torch.manual_seed(123456)
np.random.seed(123456)

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="burgers train PyTorch")
    parser.add_argument("--config_file_path", type=str, default="./configs/burgers.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Target device: 'cuda' or 'cpu'")
    input_args = parser.parse_args()
    return input_args

def train():
    '''Train and evaluate the pinns network'''
    # load configurations
    config = load_yaml_config(args.config_file_path)
    device = torch.device(args.device)

    # create dataset
    pde_data, ic_data, bc_data = create_training_dataset(config)
    burgers_train_dataset = BurgersDataset(pde_data, ic_data, bc_data)
    train_loader = DataLoader(burgers_train_dataset, 
                              batch_size=config["data"]["train"]["batch_size"],
                              shuffle=True, drop_last=True)
                              
    # create test dataset
    inputs, label = create_test_dataset(config["data"]["root_dir"])

    # define models and optimizers
    model = MLP_with_Residual(in_channels=config["model"]["in_channels"],
                              out_channels=config["model"]["out_channels"],
                              layers=config["model"]["layers"],
                              neurons=config["model"]["neurons"],
                              residual=config["model"]["residual"],
                              activation=config["model"]["activation"]).to(device)

    if config["model"]["load_ckpt"]:
        model.load_state_dict(torch.load(os.path.join(config["summary"]["ckpt_dir"], "model.pth")))

    # define optimizer and problem
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])
    problem = Burgers1D(model)

    epochs = config["data"]["train"]["epochs"]
    steps_per_epoch = len(train_loader)
    
    for epoch in range(1, 1 + epochs):
        local_time_beg = time.time()
        model.train()
        step_train_loss = 0.0
        
        for pde_b, ic_b, bc_b in train_loader:
            # Move data to GPU/CPU
            pde_b = torch.tensor(pde_b, dtype=torch.float32, requires_grad=True).to(device)
            ic_b = torch.tensor(ic_b, dtype=torch.float32).to(device)
            bc_b = torch.tensor(bc_b, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss = problem.get_loss(pde_b, ic_b, bc_b)
            loss.backward()
            optimizer.step()
            
            step_train_loss = loss.item()

        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds / steps_per_epoch
        print_log(f"epoch: {epoch} train loss: {step_train_loss:.7f} "
                  f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")

        # evaluation
        if epoch % config["summary"]["eval_interval_epochs"] == 0:
            eval_time_start = time.time()
            calculate_l2_error(model, inputs, label, config["data"]["train"]["batch_size"], device)
            print_log(f'evaluation time: {time.time() - eval_time_start:.3f}s')

    # visualization at the end
    visual(model, epochs=epochs, resolution=config["summary"]["visual_resolution"], device=device)


if __name__ == '__main__':
    print_log("pid:", os.getpid())
    args = parse_args()
    print_log(f"Running on {args.device.upper()}.")
    train()