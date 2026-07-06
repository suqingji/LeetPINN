"""train process in PyTorch"""
import argparse
import os
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import create_training_dataset, create_test_dataset, print_log
from src.model import MLP_with_Residual, NavierStokes2D
from src.utils import calculate_l2_error, visual

torch.manual_seed(0)
np.random.seed(0)

class MTLWeightedLoss(nn.Module):
    """Multi-task learning weighted loss with learnable parameters"""
    def __init__(self, num_losses):
        super(MTLWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        # 依照 Kendall et al. 2017: L = sum( exp(-log_var)*loss + log_var )
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="cylinder flow train in PyTorch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device: 'cuda' or 'cpu'")
    parser.add_argument("--config_file_path", type=str, default="./configs/cylinder_flow.yaml")
    return parser.parse_args()

def train(input_args):
    config = load_yaml_config(input_args.config_file_path)
    device = torch.device(input_args.device)
    print_log(f"Running on {device.type.upper()}.")

    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]
    geo_params = config["geometry"]

    # create datasets
    train_dataset = create_training_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=data_params["batch_size"], shuffle=True, drop_last=True)
    inputs, label = create_test_dataset(data_params['root_dir'])

    # input scaling
    coord_min = np.array(geo_params["coord_min"] + [geo_params["time_min"]]).astype(np.float32)
    coord_max = np.array(geo_params["coord_max"] + [geo_params["time_max"]]).astype(np.float32)
    input_center = list(0.5 * (coord_max + coord_min))
    input_scale = list(2.0 / (coord_max - coord_min))

    # model definition
    model = MLP_with_Residual(in_channels=model_params["in_channels"],
                              out_channels=model_params["out_channels"],
                              layers=model_params["num_layers"],
                              neurons=model_params["hidden_channels"],
                              residual=model_params["residual"],
                              activation=model_params["activation"],
                              input_scale=input_scale,
                              input_center=input_center).to(device)

    # 3 losses: PDE, BC, IC
    mtl = MTLWeightedLoss(num_losses=3).to(device)
    print_log("Use MTLWeightedLoss, num loss: 3")

    optimizer = torch.optim.Adam(list(model.parameters()) + list(mtl.parameters()), lr=optimizer_params["learning_rate"])
    problem = NavierStokes2D(model)

    epochs = optimizer_params["epochs"]
    steps_per_epoch = len(train_loader)
    print_log(f"number of steps_per_epochs: {steps_per_epoch}")
    
    os.makedirs(summary_params['ckpt_dir'], exist_ok=True)
    os.makedirs(summary_params['summary_dir'], exist_ok=True)

    for epoch in range(1, 1 + epochs):
        local_time_beg = time.time()
        model.train()
        step_train_loss = 0.0

        for pde_data, bc_data, bc_label, ic_data, ic_label in train_loader:
            pde_data = pde_data.to(device)
            bc_data, bc_label = bc_data.to(device), bc_label.to(device)
            ic_data, ic_label = ic_data.to(device), ic_label.to(device)

            optimizer.zero_grad()
            
            pde_loss, bc_loss, ic_loss = problem.get_loss(pde_data, bc_data, bc_label, ic_data, ic_label)
            loss = mtl([pde_loss, bc_loss, ic_loss])
            
            loss.backward()
            optimizer.step()
            
            step_train_loss = loss.item()

        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds / steps_per_epoch
        print_log(f"epoch: {epoch} train loss: {step_train_loss:.5f} "
                  f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")

        # evaluation
        if epoch % summary_params["test_interval"] == 0:
            eval_time_start = time.time()
            calculate_l2_error(model, inputs, label, model_params, data_params["batch_size"], device)
            print_log(f'evaluation time: {time.time() - eval_time_start:.3f}s')

        # save checkpoint
        if epoch % summary_params["save_ckpt_interval"] == 0:
            ckpt_name = f"ns_cylinder_flow-{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(summary_params['ckpt_dir'], ckpt_name))
            
    # Final Visual
    visual(model, epochs=epochs, input_data=inputs, label=label, device=device, path=summary_params['summary_dir'])

if __name__ == '__main__':
    print_log("pid:", os.getpid())
    args = parse_args()
    train(args)