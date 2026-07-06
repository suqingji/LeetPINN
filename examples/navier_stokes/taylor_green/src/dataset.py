import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_test_dataset(config):
    """ generate evaluation dataset by analytical solution"""
    coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
    coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)

    axis_x = np.linspace(coord_min[0], coord_max[0], num=100, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=100, endpoint=True)
    axis_t = np.linspace(coord_min[2], coord_max[2], num=10, endpoint=True)

    mesh_x, mesh_t, mesh_y = np.meshgrid(axis_x, axis_t, axis_y)

    inputs = np.hstack(
        (mesh_x.flatten()[:, None], mesh_y.flatten()[:, None], mesh_t.flatten()[:, None])
    ).astype(np.float32)

    label = []
    for p in inputs:
        x, y, t = p[0], p[1], p[2]
        u = - np.cos(x) * np.sin(y) * np.exp(-2 * t)
        v = np.sin(x) * np.cos(y) * np.exp(-2 * t)
        p_val = -0.25 * (np.cos(2*x) + np.cos(2*y)) * np.exp(-4*t)
        label.append(np.float32([u, v, p_val]))
    label = np.array(label)

    inputs = inputs.reshape((10, 100, 100, 3))
    label = label.reshape((10, 100, 100, 3))

    return inputs, label

class TaylorGreenDataset(Dataset):
    def __init__(self, pde_data, ic_data, bc_data):
        self.pde_data = torch.tensor(pde_data, dtype=torch.float32)
        self.ic_data = torch.tensor(ic_data, dtype=torch.float32)
        self.bc_data = torch.tensor(bc_data, dtype=torch.float32)
        # 以最大的数据集长度为准
        self.length = max(len(self.pde_data), len(self.ic_data), len(self.bc_data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 使用模运算保证不同大小的采样点能够循环匹配
        pde = self.pde_data[idx % len(self.pde_data)]
        ic = self.ic_data[idx % len(self.ic_data)]
        bc = self.bc_data[idx % len(self.bc_data)]
        return pde, ic, bc

def create_training_dataset(config):
    """create training dataset by uniform sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]
    
    x_min, y_min = geom_config["coord_min"]
    x_max, y_max = geom_config["coord_max"]
    t_min = geom_config["time_min"]
    t_max = geom_config["time_max"]

    # 1. 内部区域采样 (Domain)
    n_domain = data_config["domain"]["size"]
    domain_x = np.random.uniform(x_min, x_max, (n_domain, 1))
    domain_y = np.random.uniform(y_min, y_max, (n_domain, 1))
    domain_t = np.random.uniform(t_min, t_max, (n_domain, 1))
    pde_data = np.hstack((domain_x, domain_y, domain_t))

    # 2. 初始条件采样 (IC, t = t_min)
    n_ic = data_config["IC"]["size"]
    ic_x = np.random.uniform(x_min, x_max, (n_ic, 1))
    ic_y = np.random.uniform(y_min, y_max, (n_ic, 1))
    ic_t = np.ones((n_ic, 1)) * t_min
    ic_data = np.hstack((ic_x, ic_y, ic_t))

    # 3. 边界条件采样 (BC, 矩形四周边缘)
    n_bc = data_config["BC"]["size"]
    edges = np.random.randint(0, 4, (n_bc, 1))
    bc_x = np.zeros((n_bc, 1))
    bc_y = np.zeros((n_bc, 1))
    bc_t = np.random.uniform(t_min, t_max, (n_bc, 1))
    
    for i in range(n_bc):
        if edges[i] == 0:   # Left
            bc_x[i], bc_y[i] = x_min, np.random.uniform(y_min, y_max)
        elif edges[i] == 1: # Right
            bc_x[i], bc_y[i] = x_max, np.random.uniform(y_min, y_max)
        elif edges[i] == 2: # Bottom
            bc_x[i], bc_y[i] = np.random.uniform(x_min, x_max), y_min
        elif edges[i] == 3: # Top
            bc_x[i], bc_y[i] = np.random.uniform(x_min, x_max), y_max
            
    bc_data = np.hstack((bc_x, bc_y, bc_t))

    dataset = TaylorGreenDataset(pde_data, ic_data, bc_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=data_config["train"]["batch_size"], 
        shuffle=True, 
        drop_last=True
    )
    return dataloader