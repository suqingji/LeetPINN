import math
import torch
from torch.utils.data import Dataset, DataLoader

class PoissonDataset(Dataset):
    """Create dataset for PINN."""
    def __init__(self, geom_name, config, n_samps=None):
        self.geom_name = geom_name
        self.config = config
        self.size = n_samps if n_samps else config["data"]["domain"]["size"]
        self.bc_size = n_samps if n_samps else config["data"]["BC"]["size"]
        
        if geom_name == "interval": self.n_dim = 1
        elif geom_name in ["rectangle", "disk", "triangle", "pentagon", "polygon"]: self.n_dim = 2
        else: self.n_dim = 3

        # 生成预采样数据点
        self.domain_data = self._sample_domain(self.size)
        self.bc_data = self._sample_bc(self.bc_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.domain_data[idx], self.bc_data[idx]

    def _sample_domain(self, size):
        if self.geom_name == "rectangle":
            c_min = torch.tensor(self.config["geometry"]["rectangle"]["coord_min"])
            c_max = torch.tensor(self.config["geometry"]["rectangle"]["coord_max"])
            return torch.rand(size, 2) * (c_max - c_min) + c_min
        elif self.geom_name == "disk":
            center = torch.tensor(self.config["geometry"]["disk"]["center"])
            r = self.config["geometry"]["disk"]["radius"] * torch.sqrt(torch.rand(size, 1))
            theta = 2 * math.pi * torch.rand(size, 1)
            x = r * torch.cos(theta) + center[0]
            y = r * torch.sin(theta) + center[1]
            return torch.cat([x, y], dim=1)
        # 默认回退生成
        return torch.rand(size, self.n_dim)

    def _sample_bc(self, size):
        if self.geom_name == "rectangle":
            pts = torch.rand(size, 2)
            edges = torch.randint(0, 4, (size,))
            pts[edges == 0, 0] = 0.0
            pts[edges == 1, 0] = 1.0
            pts[edges == 2, 1] = 0.0
            pts[edges == 3, 1] = 1.0
            return pts
        elif self.geom_name == "disk":
            center = torch.tensor(self.config["geometry"]["disk"]["center"])
            r = self.config["geometry"]["disk"]["radius"]
            theta = 2 * math.pi * torch.rand(size, 1)
            x = r * torch.cos(theta) + center[0]
            y = r * torch.sin(theta) + center[1]
            return torch.cat([x, y], dim=1)
        return torch.rand(size, self.n_dim)

def create_dataset(geom_name, config, n_samps=None):
    ds = PoissonDataset(geom_name, config, n_samps)
    batch_size = config["data"]["train"]["batch_size"] if n_samps is None else n_samps
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, ds.n_dim