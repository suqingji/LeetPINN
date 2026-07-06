"""2d darcy problem and network in PyTorch"""
import numpy as np
import torch
import torch.nn as nn

class FCSequential(nn.Module):
    def __init__(self, in_channels, out_channels, layers, neurons, residual=True, act="tanh", weight_init="TruncatedNormal"):
        super(FCSequential, self).__init__()
        self.residual = residual
        self.act = nn.Tanh() if act.lower() == 'tanh' else nn.ReLU()
        
        self.input_layer = nn.Linear(in_channels, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers - 1)])
        self.output_layer = nn.Linear(neurons, out_channels)
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if weight_init == "TruncatedNormal":
                    nn.init.trunc_normal_(m.weight, std=0.01)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.act(layer(x))
            x = x + out if self.residual else out
        return self.output_layer(x)

class Darcy2D:
    r"""The steady-state 2D Darcy flow problem based on PyTorch"""
    def __init__(self, model):
        self.model = model
        self.mse = nn.MSELoss()
        self.pi = np.pi

    def force_function(self, x, y):
        return 8 * self.pi**2 * torch.sin(2 * self.pi * x) * torch.cos(2 * self.pi * y)

    def get_loss(self, pde_data, bc_data):
        # --- PDE Loss ---
        pde_data.requires_grad_(True)
        out = self.model(pde_data)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        x, y = pde_data[:, 0:1], pde_data[:, 1:2]

        du = torch.autograd.grad(u, pde_data, torch.ones_like(u), create_graph=True)[0]
        dv = torch.autograd.grad(v, pde_data, torch.ones_like(v), create_graph=True)[0]
        dp = torch.autograd.grad(p, pde_data, torch.ones_like(p), create_graph=True)[0]

        u_x, v_y = du[:, 0:1], dv[:, 1:2]
        p_x, p_y = dp[:, 0:1], dp[:, 1:2]

        loss_1 = u_x + v_y - self.force_function(x, y)
        loss_2 = u + p_x
        loss_3 = v + p_y

        pde_loss = self.mse(loss_1, torch.zeros_like(loss_1)) + \
                   self.mse(loss_2, torch.zeros_like(loss_2)) + \
                   self.mse(loss_3, torch.zeros_like(loss_3))

        # --- Boundary Condition Loss ---
        bc_out = self.model(bc_data)
        u_bc, v_bc, p_bc = bc_out[:, 0:1], bc_out[:, 1:2], bc_out[:, 2:3]
        x_bc, y_bc = bc_data[:, 0:1], bc_data[:, 1:2]

        u_boundary = -2 * self.pi * torch.cos(2 * self.pi * x_bc) * torch.cos(2 * self.pi * y_bc)
        v_boundary = 2 * self.pi * torch.sin(2 * self.pi * x_bc) * torch.sin(2 * self.pi * y_bc)
        p_boundary = torch.sin(2 * self.pi * x_bc) * torch.cos(2 * self.pi * y_bc)

        bc_loss = self.mse(u_bc, u_boundary) + \
                  self.mse(v_bc, v_boundary) + \
                  self.mse(p_bc, p_boundary)

        return pde_loss + bc_loss