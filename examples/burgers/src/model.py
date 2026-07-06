"""Burgers1D Model and Loss defined in PyTorch"""
import numpy as np
import torch
import torch.nn as nn

class MLP_with_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, layers, neurons, activation='tanh', residual=True):
        super(MLP_with_Residual, self).__init__()
        self.residual = residual
        self.act = nn.Tanh() if activation.lower() == 'tanh' else nn.ReLU()
        
        self.input_layer = nn.Linear(in_channels, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers - 1)])
        self.output_layer = nn.Linear(neurons, out_channels)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.act(layer(x))
            if self.residual:
                x = x + out
            else:
                x = out
        x = self.output_layer(x)
        return x

class Burgers1D:
    r"""Burgers 1-D problem based on PyTorch"""
    def __init__(self, model, nu=0.01/np.pi):
        self.model = model
        self.mse = nn.MSELoss()
        self.nu = nu # 粘性系数，Burgers的标准baseline常设置为 0.01/pi

    def get_loss(self, pde_data, ic_data, bc_data):
        # 1. PDE Loss (Governing Equation)
        pde_data.requires_grad_(True)
        u = self.model(pde_data)
        
        # 计算一阶导数: u_x 和 u_t
        grad_u = torch.autograd.grad(outputs=u, inputs=pde_data, 
                                     grad_outputs=torch.ones_like(u),
                                     create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]
        
        # 计算二阶导数: u_xx
        grad_u_x = torch.autograd.grad(outputs=u_x, inputs=pde_data, 
                                       grad_outputs=torch.ones_like(u_x),
                                       create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]
        
        # Burgers equation: u_t + u * u_x - nu * u_xx = 0
        pde_res = u_t + u * u_x - self.nu * u_xx
        pde_loss = self.mse(pde_res, torch.zeros_like(pde_res))

        # 2. IC Loss (Initial Condition: u = -sin(pi * x))
        u_ic = self.model(ic_data)
        x_ic = ic_data[:, 0:1]
        ic_target = -torch.sin(np.pi * x_ic)
        ic_loss = self.mse(u_ic, ic_target)

        # 3. BC Loss (Boundary Condition: u = 0)
        u_bc = self.model(bc_data)
        bc_loss = self.mse(u_bc, torch.zeros_like(u_bc))

        return pde_loss + ic_loss + bc_loss