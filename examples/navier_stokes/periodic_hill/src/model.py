import torch
import torch.nn as nn

class FCSequential(nn.Module):
    """PyTorch equivalent of MindFlow's FCSequential"""
    def __init__(self, in_channels, out_channels, layers, neurons, residual=False, act='tanh'):
        super(FCSequential, self).__init__()
        self.residual = residual
        self.act = nn.Tanh() if act == 'tanh' else nn.ReLU()
        
        self.in_layer = nn.Linear(in_channels, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers - 1)])
        self.out_layer = nn.Linear(neurons, out_channels)

    def forward(self, x):
        x = self.act(self.in_layer(x))
        for layer in self.hidden_layers:
            if self.residual:
                x = x + self.act(layer(x))
            else:
                x = self.act(layer(x))
        return self.out_layer(x)

class NavierStokesRANS:
    r"""Reynold-Averaged NavierStokes equation problem (PyTorch version)"""
    def __init__(self, model, re=5600.0, rho=1.0):
        self.model = model
        self.vis = 1.0 / re
        self.rho = rho
        self.mse_loss = nn.MSELoss()

    def get_loss(self, pde_data, bc_data, bc_label):
        # ================= 1. PDE Loss (RANS Equations) =================
        pde_data.requires_grad_(True)
        preds = self.model(pde_data)
        
        u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        uu, uv, vv = preds[:, 3:4], preds[:, 4:5], preds[:, 5:6]

        # 一阶导数
        grad_u = torch.autograd.grad(u.sum(), pde_data, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), pde_data, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), pde_data, create_graph=True)[0]
        grad_uu = torch.autograd.grad(uu.sum(), pde_data, create_graph=True)[0]
        grad_uv = torch.autograd.grad(uv.sum(), pde_data, create_graph=True)[0]
        grad_vv = torch.autograd.grad(vv.sum(), pde_data, create_graph=True)[0]

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        uu_x = grad_uu[:, 0:1]
        uv_x, uv_y = grad_uv[:, 0:1], grad_uv[:, 1:2]
        vv_y = grad_vv[:, 1:2]

        # 二阶导数
        grad_u_x = torch.autograd.grad(u_x.sum(), pde_data, create_graph=True)[0]
        grad_u_y = torch.autograd.grad(u_y.sum(), pde_data, create_graph=True)[0]
        grad_v_x = torch.autograd.grad(v_x.sum(), pde_data, create_graph=True)[0]
        grad_v_y = torch.autograd.grad(v_y.sum(), pde_data, create_graph=True)[0]

        u_xx, u_yy = grad_u_x[:, 0:1], grad_u_y[:, 1:2]
        v_xx, v_yy = grad_v_x[:, 0:1], grad_v_y[:, 1:2]

        # 控制方程残差计算
        momentum_x = u * u_x + v * u_y + (1.0/self.rho) * p_x - self.vis * (u_xx + u_yy) + uu_x + uv_y
        momentum_y = u * v_x + v * v_y + (1.0/self.rho) * p_y - self.vis * (v_xx + v_yy) + vv_y + uv_x
        continuity = u_x + v_y

        pde_res = torch.cat([momentum_x, momentum_y, continuity], dim=1)
        pde_loss = self.mse_loss(pde_res, torch.zeros_like(pde_res))

        # ================= 2. BC Loss =================
        bc_preds = self.model(bc_data)
        bc_loss = self.mse_loss(bc_preds, bc_label)

        return pde_loss + bc_loss