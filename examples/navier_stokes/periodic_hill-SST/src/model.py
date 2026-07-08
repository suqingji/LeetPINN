import torch
import torch.nn as nn
import numpy as np

class FCSequential(nn.Module):
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
    r"""RANS with k-omega SST Turbulence Model (PyTorch version)"""
    def __init__(self, model, re=5600.0, rho=1.0):
        self.model = model
        self.nu = 1.0 / re
        self.rho = rho
        self.mse_loss = nn.MSELoss()

        # K-Omega SST Parameters
        self.beta_star = 0.09
        self.kappa = 0.41
        self.a1 = 0.31

        # Inner loop (k-omega)
        self.sigma_k1 = 0.85
        self.sigma_w1 = 0.5
        self.beta1 = 0.075
        self.gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * (self.kappa**2) / np.sqrt(self.beta_star)

        # Outer loop (k-epsilon)
        self.sigma_k2 = 1.0
        self.sigma_w2 = 0.856
        self.beta2 = 0.0828
        self.gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * (self.kappa**2) / np.sqrt(self.beta_star)

    def get_loss(self, pde_data, bc_data, bc_label):
        # ================= 1. PDE Loss (SST Equations) =================
        pde_data.requires_grad_(True)
        preds = self.model(pde_data)
        
        # 提取变量并截断极小值，防止 SST 混合函数中的除零错误及 NaN
        u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        k = torch.clamp(preds[:, 3:4], min=1e-5)
        omega = torch.clamp(preds[:, 4:5], min=1e-4)

        # ---------------- 一阶导数 ----------------
        grad_u = torch.autograd.grad(u.sum(), pde_data, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), pde_data, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), pde_data, create_graph=True)[0]
        grad_k = torch.autograd.grad(k.sum(), pde_data, create_graph=True)[0]
        grad_w = torch.autograd.grad(omega.sum(), pde_data, create_graph=True)[0]

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        k_x, k_y = grad_k[:, 0:1], grad_k[:, 1:2]
        w_x, w_y = grad_w[:, 0:1], grad_w[:, 1:2]

        # ---------------- 物理衍生项计算 ----------------
        # 应变率张量大小 S
        S2 = 2.0 * u_x**2 + 2.0 * v_y**2 + (u_y + v_x)**2
        S = torch.sqrt(S2 + 1e-8)

        # 壁面距离 d 近似 (以 y=0 及 y=3.036 作为周期山上下边界近似)
        y_coord = pde_data[:, 1:2]
        d = torch.clamp(torch.min(y_coord, 3.036 - y_coord), min=1e-3)

        # 交叉扩散项 CD_kw
        cross_diff = 2.0 * self.sigma_w2 * (1.0 / omega) * (k_x * w_x + k_y * w_y)
        CD_kw = torch.clamp(cross_diff, min=1e-10)

        # 混合函数 F1
        arg1_1 = torch.sqrt(k) / (self.beta_star * omega * d)
        arg1_2 = 500.0 * self.nu / (d**2 * omega)
        arg1_3 = 4.0 * self.sigma_w2 * k / (CD_kw * d**2)
        arg1 = torch.min(torch.max(arg1_1, arg1_2), arg1_3)
        F1 = torch.tanh(arg1**4)

        # 混合函数 F2
        arg2_1 = 2.0 * torch.sqrt(k) / (self.beta_star * omega * d)
        arg2_2 = 500.0 * self.nu / (d**2 * omega)
        arg2 = torch.max(arg2_1, arg2_2)
        F2 = torch.tanh(arg2**2)

        # 动态混合系数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2
        gamma = F1 * self.gamma1 + (1.0 - F1) * self.gamma2

        # 涡粘度 (Boussinesq Eddy Viscosity)
        nu_t = self.a1 * k / torch.max(self.a1 * omega, S * F2)
        nu_eff = self.nu + nu_t
        nu_k = self.nu + sigma_k * nu_t
        nu_w = self.nu + sigma_w * nu_t

        # 湍动能生成项 Pk 限制
        Pk = torch.min(nu_t * S2, 10.0 * self.beta_star * k * omega)

        # ---------------- 二阶导数 (应力与扩散散度) ----------------
        # 动量方程应力项
        tau_xx = 2.0 * nu_eff * u_x - (2.0 / 3.0) * k
        tau_yy = 2.0 * nu_eff * v_y - (2.0 / 3.0) * k
        tau_xy = nu_eff * (u_y + v_x)

        grad_tau_xx = torch.autograd.grad(tau_xx.sum(), pde_data, create_graph=True)[0]
        grad_tau_yy = torch.autograd.grad(tau_yy.sum(), pde_data, create_graph=True)[0]
        grad_tau_xy = torch.autograd.grad(tau_xy.sum(), pde_data, create_graph=True)[0]

        tau_xx_x, tau_yy_y = grad_tau_xx[:, 0:1], grad_tau_yy[:, 1:2]
        tau_xy_x, tau_xy_y = grad_tau_xy[:, 0:1], grad_tau_xy[:, 1:2]

        # 湍流扩散通量
        flux_kx, flux_ky = nu_k * k_x, nu_k * k_y
        flux_wx, flux_wy = nu_w * w_x, nu_w * w_y

        grad_flux_kx = torch.autograd.grad(flux_kx.sum(), pde_data, create_graph=True)[0]
        grad_flux_ky = torch.autograd.grad(flux_ky.sum(), pde_data, create_graph=True)[0]
        grad_flux_wx = torch.autograd.grad(flux_wx.sum(), pde_data, create_graph=True)[0]
        grad_flux_wy = torch.autograd.grad(flux_wy.sum(), pde_data, create_graph=True)[0]

        div_flux_k = grad_flux_kx[:, 0:1] + grad_flux_ky[:, 1:2]
        div_flux_w = grad_flux_wx[:, 0:1] + grad_flux_wy[:, 1:2]

        # ---------------- 方程残差构建 ----------------
        momentum_x = u * u_x + v * u_y + (1.0 / self.rho) * p_x - tau_xx_x - tau_xy_y
        momentum_y = u * v_x + v * v_y + (1.0 / self.rho) * p_y - tau_xy_x - tau_yy_y
        continuity = u_x + v_y

        k_eq = u * k_x + v * k_y - Pk + self.beta_star * k * omega - div_flux_k
        w_eq = u * w_x + v * w_y - gamma * S2 + beta * omega**2 - div_flux_w - (1.0 - F1) * cross_diff

        pde_res = torch.cat([momentum_x, momentum_y, continuity, k_eq, w_eq], dim=1)
        pde_loss = self.mse_loss(pde_res, torch.zeros_like(pde_res))

        # ================= 2. BC Loss =================
        bc_preds = self.model(bc_data)
        u_pred, v_pred, p_pred = bc_preds[:, 0:1], bc_preds[:, 1:2], bc_preds[:, 2:3]
        k_pred = bc_preds[:, 3:4]

        # 从数据集提取标签 [u, v, p, uu, uv, vv] (索引 0, 1, 2, 3, 4, 5)
        u_true, v_true, p_true = bc_label[:, 0:1], bc_label[:, 1:2], bc_label[:, 2:3]
        
        # 使用 2D 正应力近似计算边界湍动能 k
        uu_true, vv_true = bc_label[:, 3:4], bc_label[:, 5:6]
        k_true = 0.5 * (uu_true + vv_true)

        bc_res = torch.cat([u_pred - u_true, v_pred - v_true, p_pred - p_true, k_pred - k_true], dim=1)
        bc_loss = self.mse_loss(bc_res, torch.zeros_like(bc_res))

        return pde_loss + bc_loss