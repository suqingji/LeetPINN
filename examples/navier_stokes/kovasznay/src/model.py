import math
import torch
import torch.nn as nn

class FCSequential(nn.Module):
    """PyTorch equivalent of MindFlow's FCSequential"""
    def __init__(self, in_channels, out_channels, layers, neurons, residual):
        super(FCSequential, self).__init__()
        self.residual = residual
        
        self.in_layer = nn.Linear(in_channels, neurons)
        self.act = nn.Tanh()
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for _ in range(layers - 1)]
        )
        
        self.out_layer = nn.Linear(neurons, out_channels)

    def forward(self, x):
        x = self.act(self.in_layer(x))
        for layer in self.hidden_layers:
            if self.residual:
                x = x + self.act(layer(x))
            else:
                x = self.act(layer(x))
        return self.out_layer(x)

class Kovasznay:
    """Define the loss of the Kovasznay equation."""
    def __init__(self, model, re=20):
        self.model = model
        self.re = re
        self.nu = 1.0 / self.re
        self.l = 1.0 / (2 * self.nu) - math.sqrt(
            1.0 / (4 * self.nu**2) + 4 * math.pi**2
        )
        self.mse_loss = nn.MSELoss()

    def u_func(self, x, y):
        return 1.0 - torch.exp(self.l * x) * torch.cos(2 * math.pi * y)

    def v_func(self, x, y):
        return (self.l / (2 * math.pi)) * torch.exp(self.l * x) * torch.sin(2 * math.pi * y)

    def p_func(self, x, y):
        return 0.5 * (1.0 - torch.exp(2 * self.l * x))

    def get_loss(self, pde_data, bc_data):
        # ================= 1. PDE Loss =================
        pde_data.requires_grad_(True)
        preds = self.model(pde_data)
        u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

        # 一阶导数
        grad_u = torch.autograd.grad(u.sum(), pde_data, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), pde_data, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), pde_data, create_graph=True)[0]
        
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        # 二阶导数
        grad_u_x = torch.autograd.grad(u_x.sum(), pde_data, create_graph=True)[0]
        grad_u_y = torch.autograd.grad(u_y.sum(), pde_data, create_graph=True)[0]
        grad_v_x = torch.autograd.grad(v_x.sum(), pde_data, create_graph=True)[0]
        grad_v_y = torch.autograd.grad(v_y.sum(), pde_data, create_graph=True)[0]
        
        u_xx, u_yy = grad_u_x[:, 0:1], grad_u_y[:, 1:2]
        v_xx, v_yy = grad_v_x[:, 0:1], grad_v_y[:, 1:2]
        
        momentum_x = u * u_x + v * u_y + p_x - (1.0 / self.re) * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - (1.0 / self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        
        pde_res = torch.cat([momentum_x, momentum_y, continuity], dim=1)
        pde_loss = self.mse_loss(pde_res, torch.zeros_like(pde_res))

        # ================= 2. BC Loss =================
        bc_preds = self.model(bc_data)
        x_bc, y_bc = bc_data[:, 0:1], bc_data[:, 1:2]
        
        u_bc_true = self.u_func(x_bc, y_bc)
        v_bc_true = self.v_func(x_bc, y_bc)
        p_bc_true = self.p_func(x_bc, y_bc)
        
        bc_res_u = bc_preds[:, 0:1] - u_bc_true
        bc_res_v = bc_preds[:, 1:2] - v_bc_true
        bc_res_p = bc_preds[:, 2:3] - p_bc_true
        
        bc_res = torch.cat([bc_res_u, bc_res_v, bc_res_p], dim=1)
        bc_loss = self.mse_loss(bc_res, torch.zeros_like(bc_res))

        return pde_loss + bc_loss