"""create dataset in PyTorch"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BurgersDataset(Dataset):
    def __init__(self, pde_data, ic_data, bc_data):
        self.pde_data = pde_data
        self.ic_data = ic_data
        self.bc_data = bc_data

    def __len__(self):
        return len(self.pde_data)

    def __getitem__(self, idx):
        return self.pde_data[idx], self.ic_data[idx], self.bc_data[idx]

def create_training_dataset(config):
    """create training dataset by online sampling (PyTorch version)"""
    geom_config = config["geometry"]
    data_config = config["data"]

    t_min, t_max = geom_config["time_min"], geom_config["time_max"]
    x_min, x_max = geom_config["coord_min"], geom_config["coord_max"]

    domain_size = data_config["domain"]["size"]
    ic_size = data_config["IC"]["size"]
    bc_size = data_config["BC"]["size"]

    # Domain (PDE) Data
    x_domain = np.random.uniform(x_min, x_max, (domain_size, 1))
    t_domain = np.random.uniform(t_min, t_max, (domain_size, 1))
    pde_data = np.hstack((x_domain, t_domain))

    # IC Data (t = 0)
    x_ic = np.random.uniform(x_min, x_max, (ic_size, 1))
    t_ic = np.zeros((ic_size, 1))
    ic_data = np.hstack((x_ic, t_ic))

    # BC Data (x = -1 or x = 1)
    x_bc = np.random.choice([x_min, x_max], (bc_size, 1))
    t_bc = np.random.uniform(t_min, t_max, (bc_size, 1))
    bc_data = np.hstack((x_bc, t_bc))

    return pde_data, ic_data, bc_data

def create_test_dataset(test_dataset_path):
    test_data = np.load(os.path.join(test_dataset_path, "Burgers.npz"))
    x, t, u = test_data["x"], test_data["t"], test_data["usol"].T
    xx, tt = np.meshgrid(x, t)

    test_data = np.vstack((np.ravel(xx), np.ravel(tt))).T
    test_label = u.flatten()[:, None]
    
    return test_data, test_label