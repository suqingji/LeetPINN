import torch
import torch.nn as nn

class PINNModel(nn.Module):
    """等价于 MindFlow 中的 MultiScaleFCSequential"""
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
        # 输入标准化
        x = (x - center) * scale
        return self.net(x)


class InvNavierStokes:
    """计算 Navier-Stokes 的 PDE 损失和 Data 损失"""
    def __init__(self, model, theta):
        self.model = model
        self.theta = theta  # 包含 theta1 和 theta2 的 Parameter
        self.mse_loss = nn.MSELoss()

    def get_loss(self, pde_data, label):
        # 必须开启梯度追踪以计算高阶导数
        pde_data.requires_grad_(True)
        
        # 预测 U, V, P
        preds = self.model(pde_data)
        u = preds[:, 0:1]
        v = preds[:, 1:2]
        p = preds[:, 2:3]
        
        # 计算 data loss
        data_loss = self.mse_loss(preds, label)
        
        # ================= PDE Loss (Autograd) =================
        # 求解 u 对 x, y, t 的一阶导
        grad_u = torch.autograd.grad(u, pde_data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]
        
        # 求解 v 对 x, y, t 的一阶导
        grad_v = torch.autograd.grad(v, pde_data, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        v_t = grad_v[:, 2:3]
        
        # 求解 p 对 x, y 的一阶导
        grad_p = torch.autograd.grad(p, pde_data, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        
        # 求解 u 对 x, y 的二阶导
        u_xx = torch.autograd.grad(u_x, pde_data, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, pde_data, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        # 求解 v 对 x, y 的二阶导
        v_xx = torch.autograd.grad(v_x, pde_data, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, pde_data, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        theta1 = self.theta[0]
        theta2 = self.theta[1]
        
        # 构建物理方程残差
        momentum_x = u_t + theta1 * (u * u_x + v * u_y) + p_x - theta2 * (u_xx + u_yy)
        momentum_y = v_t + theta1 * (u * v_x + v * v_y) + p_y - theta2 * (v_xx + v_yy)
        continuity = u_x + v_y
        
        pde_res = torch.cat([momentum_x, momentum_y, continuity], dim=1)
        pde_loss = self.mse_loss(pde_res, torch.zeros_like(pde_res))
        
        return pde_loss + data_loss