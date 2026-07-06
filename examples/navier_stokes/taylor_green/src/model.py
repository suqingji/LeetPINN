import torch
import torch.nn as nn

class PINNModel(nn.Module):
    """Equivalent to MultiScaleFCSequential in MindFlow"""
    def __init__(self, in_channels, out_channels, layers, neurons, input_scale, input_center):
        super(PINNModel, self).__init__()
        self.input_scale = torch.tensor(input_scale, dtype=torch.float32)
        self.input_center = torch.tensor(input_center, dtype=torch.float32)
        
        net_layers = []
        net_layers.append(nn.Linear(in_channels, neurons))
        net_layers.append(nn.Tanh())
        
        for _ in range(layers - 1):
            net_layers.append(nn.Linear(neurons, neurons))
            net_layers.append(nn.Tanh())
            
        net_layers.append(nn.Linear(neurons, out_channels))
        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        scale = self.input_scale.to(x.device)
        center = self.input_center.to(x.device)
        x = (x - center) * scale
        return self.net(x)

class NavierStokes2D:
    def __init__(self, model, re=100.0):
        self.model = model
        self.re = re
        self.mse_loss = nn.MSELoss()

    def get_loss(self, pde_data, ic_data, bc_data):
        # ================= 1. PDE Loss (Navier-Stokes) =================
        pde_data.requires_grad_(True)
        preds = self.model(pde_data)
        u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        
        # 一阶导数
        grad_u = torch.autograd.grad(u.sum(), pde_data, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), pde_data, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), pde_data, create_graph=True)[0]
        
        u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        # 二阶导数
        grad_u_x = torch.autograd.grad(u_x.sum(), pde_data, create_graph=True)[0]
        grad_u_y = torch.autograd.grad(u_y.sum(), pde_data, create_graph=True)[0]
        grad_v_x = torch.autograd.grad(v_x.sum(), pde_data, create_graph=True)[0]
        grad_v_y = torch.autograd.grad(v_y.sum(), pde_data, create_graph=True)[0]
        
        u_xx, u_yy = grad_u_x[:, 0:1], grad_u_y[:, 1:2]
        v_xx, v_yy = grad_v_x[:, 0:1], grad_v_y[:, 1:2]
        
        # 控制方程残差
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1.0 / self.re) * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1.0 / self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        
        pde_res = torch.cat([momentum_x, momentum_y, continuity], dim=1)
        loss_pde = self.mse_loss(pde_res, torch.zeros_like(pde_res))

        # ================= 2. IC Loss =================
        pred_ic = self.model(ic_data)
        x_ic, y_ic = ic_data[:, 0:1], ic_data[:, 1:2]
        u_ic_true = -torch.cos(x_ic) * torch.sin(y_ic)
        v_ic_true = torch.sin(x_ic) * torch.cos(y_ic)
        p_ic_true = -0.25 * (torch.cos(2*x_ic) + torch.cos(2*y_ic))
        
        ic_res_u = pred_ic[:, 0:1] - u_ic_true
        ic_res_v = pred_ic[:, 1:2] - v_ic_true
        ic_res_p = pred_ic[:, 2:3] - p_ic_true
        ic_res = torch.cat([ic_res_u, ic_res_v, ic_res_p], dim=1)
        loss_ic = self.mse_loss(ic_res, torch.zeros_like(ic_res))

        # ================= 3. BC Loss =================
        pred_bc = self.model(bc_data)
        x_bc, y_bc, t_bc = bc_data[:, 0:1], bc_data[:, 1:2], bc_data[:, 2:3]
        u_bc_true = -torch.cos(x_bc) * torch.sin(y_bc) * torch.exp(-2*t_bc)
        v_bc_true = torch.sin(x_bc) * torch.cos(y_bc) * torch.exp(-2*t_bc)
        p_bc_true = -0.25 * (torch.cos(2*x_bc) + torch.cos(2*y_bc)) * torch.exp(-4*t_bc)
        
        bc_res_u = pred_bc[:, 0:1] - u_bc_true
        bc_res_v = pred_bc[:, 1:2] - v_bc_true
        bc_res_p = pred_bc[:, 2:3] - p_bc_true
        bc_res = torch.cat([bc_res_u, bc_res_v, bc_res_p], dim=1)
        loss_bc = self.mse_loss(bc_res, torch.zeros_like(bc_res))

        return loss_pde + loss_ic + loss_bc