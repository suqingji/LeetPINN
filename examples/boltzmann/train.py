# train.py
import time
import argparse
import numpy as np
import yaml
import torch

from src.boltzmann import (
    BoltzmannBGK, BoltzmannFBGK, BoltzmannFSM, BoltzmannLA, BoltzmannLR, get_reduced_kernel
)
from src.utils import get_vdis, get_vtuple, visual, get_potential, get_mu, get_kn_bzm, valid_model, save_points
from src.cells import SplitNet, SplitNetLR
from src.dataset import Wave1DDataset

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description="boltzmann train (PyTorch)")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--config_file_path", type=str, default="./config/WaveD1V3_BGK.yaml")

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train():
    args = parser.parse_args()
    config = load_yaml_config(args.config_file_path)
    device = torch.device(args.device)

    dataset = Wave1DDataset(config, device=device)
    vdis, _ = get_vdis(config["vmesh"])
    vt, _ = get_vtuple(config["vmesh"])
    
    # 强制将张量放置到目标设备
    vdis = vdis.to(device)
    vt = tuple([v.to(device) for v in vt])

    if config["collision"] == "BGK":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis).to(device)
        problem = BoltzmannBGK(model, config["kn"], config["vmesh"]).to(device)
    elif config["collision"] == "FBGK":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis).to(device)
        omega = config["omega"]
        mu_ref = get_mu(get_potential(omega), omega, config["kn"])
        problem = BoltzmannFBGK(model, config["kn"], config["vmesh"], omega=omega).to(device)
    elif config["collision"] == "FSM":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis).to(device)
        omega = config["omega"]
        mu_ref = get_mu(get_potential(omega), omega, config["kn"])
        kn_bzm = get_kn_bzm(get_potential(omega), mu_ref)
        problem = BoltzmannFSM(model, kn_bzm, config["vmesh"], omega=omega).to(device)
    elif config["collision"] == "LR":
        model = SplitNetLR(2, config["model"]["layers"], config["model"]["neurons"], vt, config["rank"]).to(device)
        problem = BoltzmannLR(model, config["kn"], config["vmesh"]).to(device)
    elif config["collision"] == "LA":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis).to(device)
        omega = config["omega"]
        mu_ref = get_mu(get_potential(omega), omega, config["kn"])
        kn_bzm = get_kn_bzm(get_potential(omega), mu_ref)
        traindata = np.load(config["approx_data"])["f"]
        kernel_f, kernel_g, kernel_k = get_reduced_kernel(config, traindata)
        problem = BoltzmannLA(model, kn_bzm, config["vmesh"], 
                              torch.tensor(kernel_f, dtype=torch.float32).to(device),
                              torch.tensor(kernel_k, dtype=torch.float32).to(device),
                              torch.tensor(kernel_g, dtype=torch.float32).to(device)).to(device)
    else:
        raise ValueError("Invalid collision model")

    optim = torch.optim.Adam(problem.parameters(), lr=config["optim"]["lr_scheduler"]["max_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config["optim"]["Adam_steps"], eta_min=config["optim"]["lr_scheduler"]["min_lr"])

    for i in range(1, config["optim"]["Adam_steps"] + 1):
        time_beg = time.time()
        optim.zero_grad()
        
        # Get dynamic sampled dataset
        domain_points, iv_points, bv_points = dataset()
        
        loss, _ = problem(domain_points, iv_points, bv_points[0], bv_points[1])
        loss.backward()
        optim.step()
        scheduler.step()

        if i % 100 == 0:
            e_sum = loss.item()
            print(f"epoch: {i} train loss: {e_sum:.3e} epoch time: {(time.time() - time_beg) * 1000 :.3f}ms")
            if config.get("ref_solution"):
                valid_model(config, problem)
                
    torch.save(problem.state_dict(), f'./model_{config["collision"]}_kn{config["kn"]}.pth')
    visual(problem, config["visual_resolution"], f'Wave_{config["collision"]}_kn{config["kn"]}.png')
    
    if config["save_points"] and (config["collision"] != "LR"):
        save_points(problem, points=1000, filename=f'{config["collision"]}_kn{config["kn"]}.npz')

if __name__ == "__main__":
    start_time = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - start_time))