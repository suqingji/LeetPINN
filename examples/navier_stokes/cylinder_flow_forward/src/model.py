"""Navier-Stokes 2D and Model defined in PyTorch"""
import numpy as np
import torch
import torch.nn as nn

class MLP_with_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, layers, neurons, activation='sin', residual=False, input_scale=None, input_center=None):
        super(MLP_with_Residual, self).__init__()
        self.residual = residual
        self.act = torch.sin if activation.lower() == 'sin' else torch.tanh
        
        # 数据归一化参数
        self.register_buffer('input_scale', torch.tensor(input_scale, dtype=torch.float32) if input_scale else torch.ones(in_channels))
        self.register_buffer('input_center', torch.tensor(input_center, dtype=torch.float32) if input_center else torch.zeros(in_channels))
        
        self.input_layer = nn.Linear(in_channels, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers - 1)])
        self.output_layer = nn.Linear(neurons, out_channels)

    def forward(self, x):
        # 坐标归一化
        x = (x - self.input_center) * self.input_scale
        
        x = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.act(layer(x))
            x = x + out if self.residual else out
        x = self.output_layer(x)
        return x

class NavierStokes2D:
    r"""2D NavierStokes equation problem based on PyTorch"""
    def __init__(self, model, re=100.0):
        self.model = model
        self.re = re
        self.mse = nn.MSELoss()

    def get_loss(self, pde_data, bc_data, bc_label, ic_data, ic_label):
        # 1. PDE Loss (Navier-Stokes Equations)
        pde_data.requires_grad_(True)
        pred = self.model(pde_data)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        
        # 计算一阶导数
        du = torch.autograd.grad(u, pde_data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dv = torch.autograd.grad(v, pde_data, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dp = torch.autograd.grad(p, pde_data, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
        p_x, p_y = dp[:, 0:1], dp[:, 1:2]
        
        # 计算二阶导数
        du_x = torch.autograd.grad(u_x, pde_data, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        du_y = torch.autograd.grad(u_y, pde_data, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        dv_x = torch.autograd.grad(v_x, pde_data, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        dv_y = torch.autograd.grad(v_y, pde_data, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        u_xx, u_yy = du_x[:, 0:1], du_y[:, 1:2]
        v_xx, v_yy = dv_x[:, 0:1], dv_y[:, 1:2]
        
        # 控制方程残差
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1.0 / self.re) * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1.0 / self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        
        pde_loss = self.mse(momentum_x, torch.zeros_like(momentum_x)) + \
                   self.mse(momentum_y, torch.zeros_like(momentum_y)) + \
                   self.mse(continuity, torch.zeros_like(continuity))

        # 2. IC Loss
        ic_pred = self.model(ic_data)
        ic_loss = self.mse(ic_pred, ic_label)

        # 3. BC Loss
        bc_pred = self.model(bc_data)
        # 根据原代码逻辑，只约束 u 和 v，由于原 MindFlow 代码是拼在一块算的，这里完全约束输出
        bc_loss = self.mse(bc_pred[:, :2], bc_label[:, :2])

        # 返回各项 loss，以便外部配合 MTL 加权
        return pde_loss, bc_loss, ic_loss