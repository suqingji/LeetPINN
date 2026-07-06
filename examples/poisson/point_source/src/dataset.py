import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PointSourceDataset(Dataset):
    def __init__(self, config):
        self.size = config['data']['domain']['size']
        
        # PDE 求解域均匀采样
        c_min = torch.tensor(config['rectangle']['coord_min'])
        c_max = torch.tensor(config['rectangle']['coord_max'])
        self.pde_data = torch.rand(self.size, 2) * (c_max - c_min) + c_min

        # 点源局部区域采样
        s_min = torch.tensor(config['rectangle_src']['coord_min'])
        s_max = torch.tensor(config['rectangle_src']['coord_max'])
        self.src_data = torch.rand(self.size, 2) * (s_max - s_min) + s_min

        # 边界(BC)采样: 从4条边上随机取点
        bc_data = torch.rand(self.size, 2) * (c_max - c_min) + c_min
        edges = torch.randint(0, 4, (self.size,))
        bc_data[edges == 0, 0] = c_min[0]
        bc_data[edges == 1, 0] = c_max[0]
        bc_data[edges == 2, 1] = c_min[1]
        bc_data[edges == 3, 1] = c_max[1]
        self.bc_data = bc_data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.pde_data[idx], self.bc_data[idx], self.src_data[idx]

def create_train_dataset(config):
    ds = PointSourceDataset(config)
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)

def create_test_dataset(config, n_samps_per_axis=100):
    axis_x = np.linspace(config['rectangle']['coord_min'][0], config['rectangle']['coord_max'][0], n_samps_per_axis, endpoint=True)
    axis_y = np.linspace(config['rectangle']['coord_min'][1], config['rectangle']['coord_max'][1], n_samps_per_axis, endpoint=True)
    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)
    mesh = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=-1)

    label = np.zeros(mesh.shape[0], dtype=np.float32)
    truncation_number = 100
    x_src, y_src = math.pi / 2, math.pi / 2
    
    # 依据二重傅立叶正弦级数计算真实解析解
    for i in range(1, truncation_number + 1):
        for j in range(1, truncation_number + 1):
            label += np.sin(i * mesh[:, 0]) * math.sin(i * x_src) * \
                     np.sin(j * mesh[:, 1]) * math.sin(j * y_src) / (i**2 + j**2)

    label = label * 4.0 / (math.pi**2)
    return torch.tensor(mesh, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)