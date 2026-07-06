"""create dataset in PyTorch"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def print_log(*args):
    print(*args)

class CylinderFlowDataset(Dataset):
    def __init__(self, pde_data, bc_data, bc_label, ic_data, ic_label):
        self.pde_data = pde_data
        self.bc_data = bc_data
        self.bc_label = bc_label
        self.ic_data = ic_data
        self.ic_label = ic_label
        # 为了兼容 DataLoader，以数据量最大的一项为基准，其他项循环取样
        self.max_len = max(len(pde_data), len(bc_data), len(ic_data))

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        idx_pde = idx % len(self.pde_data)
        idx_bc = idx % len(self.bc_data)
        idx_ic = idx % len(self.ic_data)
        return (
            self.pde_data[idx_pde], 
            self.bc_data[idx_bc], self.bc_label[idx_bc],
            self.ic_data[idx_ic], self.ic_label[idx_ic]
        )

def create_test_dataset(test_data_path):
    """load labeled data for evaluation"""
    print_log("get dataset path: {}".format(test_data_path))
    paths = [os.path.join(test_data_path, 'eval_points.npy'),
             os.path.join(test_data_path, 'eval_label.npy')]
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print_log("check eval dataset length: {}".format(inputs.shape))
    return inputs, label

def create_training_dataset(config):
    """create training dataset by online sampling (PyTorch version)"""
    geom_config = config["geometry"]
    data_config = config["data"]

    # 1. Domain (PDE) Data Sampling
    domain_size = data_config["domain"]["size"]
    x_min, x_max = geom_config["coord_min"][0], geom_config["coord_max"][0]
    y_min, y_max = geom_config["coord_min"][1], geom_config["coord_max"][1]
    t_min, t_max = geom_config["time_min"], geom_config["time_max"]

    x_domain = np.random.uniform(x_min, x_max, (domain_size, 1))
    y_domain = np.random.uniform(y_min, y_max, (domain_size, 1))
    t_domain = np.random.uniform(t_min, t_max, (domain_size, 1))
    pde_data = np.hstack((x_domain, y_domain, t_domain)).astype(np.float32)

    # 2. BC and IC Data Loading
    data_dir = data_config["root_dir"]
    print_log(f"loading boundary and initial data from: {data_dir}")
    bc_points = np.load(os.path.join(data_dir, "bc_points.npy")).astype(np.float32)
    bc_label = np.load(os.path.join(data_dir, "bc_label.npy")).astype(np.float32)
    ic_points = np.load(os.path.join(data_dir, "ic_points.npy")).astype(np.float32)
    ic_label = np.load(os.path.join(data_dir, "ic_label.npy")).astype(np.float32)

    dataset = CylinderFlowDataset(pde_data, bc_points, bc_label, ic_points, ic_label)
    return dataset