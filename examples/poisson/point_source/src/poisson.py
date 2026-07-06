import math
import torch
import torch.nn as nn

class MTLWeightedLoss(nn.Module):
    def __init__(self, num_losses=3):
        super(MTLWeightedLoss, self).__init__()
        # 初始化可学习的损失权重参数
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum

class PoissonPINN(nn.Module):
    def __init__(self, model):
        super(PoissonPINN, self).__init__()
        self.model = model
        self.alpha = 0.01  # kernel width
        self.loss_fn = MTLWeightedLoss(num_losses=3)
        self.x_src = math.pi / 2
        self.y_src = math.pi / 2

    def pde_residual(self, x_data):
        """计算给定区域内的偏微分方程残差"""
        x_data.requires_grad_(True)
        u = self.model(x_data)
        
        # 一阶导数
        u_x = torch.autograd.grad(u.sum(), x_data, create_graph=True)[0]
        # 二阶导数 (拉普拉斯)
        u_xx = torch.autograd.grad(u_x[:, 0].sum(), x_data, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_x[:, 1].sum(), x_data, create_graph=True)[0][:, 1:2]

        # 源项: 拉普拉斯概率密度函数逼近 Dirac \delta 函数
        x_dist = torch.abs(x_data[:, 0:1] - self.x_src)
        y_dist = torch.abs(x_data[:, 1:2] - self.y_src)
        force_term = (0.25 / (self.alpha ** 2)) * torch.exp(-(x_dist + y_dist) / self.alpha)

        poisson = u_xx + u_yy + force_term
        return poisson

    def forward(self, pde_data, bc_data, src_data):
        # 1. 求解域内损失
        pde_res = self.pde_residual(pde_data)
        loss_pde = torch.mean(pde_res ** 2)

        # 2. 边界条件损失 (全边界 Dirichlet u=0)
        u_bc = self.model(bc_data)
        loss_bc = torch.mean(u_bc ** 2)

        # 3. 点源局部区域内损失 (为捕获激变，加强该区域 PDE 残差监控)
        src_res = self.pde_residual(src_data)
        loss_src = torch.mean(src_res ** 2)

        total_loss = self.loss_fn((loss_pde, loss_bc, loss_src))
        return total_loss