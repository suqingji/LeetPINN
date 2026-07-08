import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PeriodicHillDataset(Dataset):
    """convert raw data into train dataset"""
    def __init__(self, bc_data, pde_data):
        self.coord = torch.tensor(bc_data[:, :2], dtype=torch.float32)
        self.label = torch.tensor(bc_data[:, 2:], dtype=torch.float32)
        self.pde_coord = torch.tensor(pde_data[:, :2], dtype=torch.float32)
        self.bc_len = self.coord.shape[0]

    def __getitem__(self, index):
        # 使用取余操作使较小的 bc_data 可以和 pde_data 匹配
        return self.pde_coord[index], self.coord[index % self.bc_len], self.label[index % self.bc_len]

    def __len__(self):
        return self.pde_coord.shape[0]

def create_test_dataset(data_path):
    """load labeled data for evaluation"""
    data = np.load(data_path)  # shape=(700*300, 10)  x, y, u, v, p, uu, uv, vv, rho, nu
    data = data.reshape((700, 300, 10)).astype(np.float32)
    data = data[:, :, :8]
    test_data = data.reshape((-1, 8))
    test_coord = test_data[:, :2]
    test_label = test_data[:, 2:]
    return test_coord, test_label

def create_train_dataset(data_path, batch_size):
    """create training dataset by online sampling"""
    data = np.load(data_path)  # shape=(700*300, 10)  x, y, u, v, p, uu, uv, vv, rho, nu
    data = np.reshape(data, (300, 700, 10)).astype(np.float32)
    data = data[:, :, :8]

    # 切片逻辑与 MindSpore 原版保持完全一致
    bc_data = data[:5].reshape((-1, 8))
    bc_data = np.concatenate((bc_data, data[-5:].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate((bc_data, data[5:-5, :5].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate((bc_data, data[5:-5, -5:].reshape((-1, 8))), axis=0)

    pde_data = data[5:-5, 5:-5].reshape((-1, 8))
    
    dataset = PeriodicHillDataset(bc_data, pde_data)
    
    # 替换 MindSpore 的 GeneratorDataset 为 PyTorch 的 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader