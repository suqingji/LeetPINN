# src/dataset.py
import torch
import torch.nn as nn
from src.cells import Maxwellian
from src.utils import mesh_nd

class Wave1DDataset:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.xmax = config["xtmesh"]["xmax"]
        self.xmin = config["xtmesh"]["xmin"]
        nv = config["vmesh"]["nv"]
        vmin = config["vmesh"]["vmin"]
        vmax = config["vmesh"]["vmax"]
        v, _ = mesh_nd(vmin, vmax, nv)
        self.vdis = torch.tensor(v, dtype=torch.float32, device=device)
        self.maxwellian = Maxwellian(self.vdis)
        
        self.iv_points = self.config["dataset"]["iv_points"]
        self.bv_points = self.config["dataset"]["bv_points"]
        self.in_points = self.config["dataset"]["in_points"]

    def __call__(self):
        iv_x = torch.rand((self.iv_points, 1), device=self.device) * (self.xmax - self.xmin) + self.xmin
        iv_t = torch.zeros_like(iv_x)

        bv_x1 = -0.5 * torch.ones((self.bv_points, 1), device=self.device)
        bv_t1 = torch.rand((self.bv_points, 1), device=self.device) * 0.1
        bv_x2 = 0.5 * torch.ones((self.bv_points, 1), device=self.device)
        bv_t2 = bv_t1

        in_x = torch.rand((self.in_points, 1), device=self.device) - 0.5
        in_t = torch.rand((self.in_points, 1), device=self.device) * 0.1

        return (
            torch.cat([in_x, in_t], dim=-1),
            torch.cat([iv_x, iv_t], dim=-1),
            (torch.cat([bv_x1, bv_t1], dim=-1), torch.cat([bv_x2, bv_t2], dim=-1))
        )