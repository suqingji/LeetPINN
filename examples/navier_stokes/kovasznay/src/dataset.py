import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

class KovasznayDataset(Dataset):
    def __init__(self, pde_data, bc_data):
        self.pde_data = torch.tensor(pde_data, dtype=torch.float32)
        self.bc_data = torch.tensor(bc_data, dtype=torch.float32)
        self.length = max(len(self.pde_data), len(self.bc_data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pde = self.pde_data[idx % len(self.pde_data)]
        bc = self.bc_data[idx % len(self.bc_data)]
        return pde, bc

def create_dataset(config, n_samps=None):
    """Create dataset by uniform random sampling."""
    if n_samps is not None:
        config = deepcopy(config)
        config["data"]["domain"]["size"] = n_samps
        config["data"]["BC"]["size"] = n_samps
        config["batch_size"] = n_samps

    n_domain = config["data"]["domain"]["size"]
    n_bc = config["data"]["BC"]["size"]
    batch_size = config["batch_size"]

    x_min, y_min = config["geometry"]["rectangle"]["coord_min"]
    x_max, y_max = config["geometry"]["rectangle"]["coord_max"]

    # 1. Domain Sampling (内部区域)
    domain_x = np.random.uniform(x_min, x_max, (n_domain, 1))
    domain_y = np.random.uniform(y_min, y_max, (n_domain, 1))
    pde_data = np.hstack((domain_x, domain_y))

    # 2. BC Sampling (四个边界)
    n_bc_per_edge = n_bc // 4
    
    # Bottom edge (y = y_min)
    edge1_x = np.random.uniform(x_min, x_max, (n_bc_per_edge, 1))
    edge1_y = np.full((n_bc_per_edge, 1), y_min)
    
    # Top edge (y = y_max)
    edge2_x = np.random.uniform(x_min, x_max, (n_bc_per_edge, 1))
    edge2_y = np.full((n_bc_per_edge, 1), y_max)
    
    # Left edge (x = x_min)
    edge3_x = np.full((n_bc_per_edge, 1), x_min)
    edge3_y = np.random.uniform(y_min, y_max, (n_bc_per_edge, 1))
    
    # Right edge (x = x_max)
    edge4_x = np.full((n_bc_per_edge, 1), x_max)
    edge4_y = np.random.uniform(y_min, y_max, (n_bc_per_edge, 1))

    bc_x = np.vstack([edge1_x, edge2_x, edge3_x, edge4_x])
    bc_y = np.vstack([edge1_y, edge2_y, edge3_y, edge4_y])
    bc_data = np.hstack((bc_x, bc_y))
    np.random.shuffle(bc_data) # 打乱边界点

    dataset = KovasznayDataset(pde_data, bc_data)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return dataloader