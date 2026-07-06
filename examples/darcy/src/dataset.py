"""create dataset in PyTorch"""
import numpy as np
from scipy.constants import pi as PI
import torch
from torch.utils.data import Dataset

class DarcyDataset(Dataset):
    def __init__(self, pde_data, bc_data):
        self.pde_data = pde_data
        self.bc_data = bc_data

    def __len__(self):
        return len(self.pde_data)

    def __getitem__(self, idx):
        return self.pde_data[idx], self.bc_data[idx % len(self.bc_data)]

def create_test_dataset(config):
    """load labeled data for evaluation"""
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    axis_size = config["geometry"]["axis_size"]

    axis_x = np.linspace(coord_min[0], coord_max[0], num=axis_size, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)
    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)

    # 按照 y, x 展平，对应原本的代码逻辑
    input_data = np.hstack((mesh_y.flatten()[:, None], mesh_x.flatten()[:, None])).astype(np.float32)

    label = np.zeros((axis_size, axis_size, 3))
    for i in range(axis_size):
        for j in range(axis_size):
            in_x = axis_x[i]
            in_y = axis_y[j]
            label[i, j, 0] = -2 * PI * np.cos(2 * PI * in_x) * np.cos(2 * PI * in_y)
            label[i, j, 1] = 2 * PI * np.sin(2 * PI * in_x) * np.sin(2 * PI * in_y)
            label[i, j, 2] = np.sin(2 * PI * in_x) * np.cos(2 * PI * in_y)

    label = label.reshape(-1, 3).astype(np.float32)
    return input_data, label

def create_training_dataset(config, name="flow_region"):
    """create training dataset by grid sampling"""
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    domain_size = config["data"]["domain"]["size"]
    bc_size = config["data"]["BC"]["size"]

    # Domain Data (Grid)
    x_dom = np.linspace(coord_min[0], coord_max[0], domain_size[0])
    y_dom = np.linspace(coord_min[1], coord_max[1], domain_size[1])
    mx, my = np.meshgrid(x_dom, y_dom)
    pde_data = np.hstack((mx.flatten()[:, None], my.flatten()[:, None])).astype(np.float32)

    # BC Data (Edges)
    points_per_edge = bc_size // 4
    edge_x = np.linspace(coord_min[0], coord_max[0], points_per_edge)
    edge_y = np.linspace(coord_min[1], coord_max[1], points_per_edge)
    
    bc_bottom = np.hstack((edge_x[:, None], np.full_like(edge_x[:, None], coord_min[1])))
    bc_top = np.hstack((edge_x[:, None], np.full_like(edge_x[:, None], coord_max[1])))
    bc_left = np.hstack((np.full_like(edge_y[:, None], coord_min[0]), edge_y[:, None]))
    bc_right = np.hstack((np.full_like(edge_y[:, None], coord_max[0]), edge_y[:, None]))
    
    bc_data = np.vstack((bc_bottom, bc_top, bc_left, bc_right)).astype(np.float32)

    return DarcyDataset(pde_data, bc_data)