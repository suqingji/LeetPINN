# src/cells.py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, vmap

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Maxwellian(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer('v', v)
        self.dim = v.shape[-1]
        self.pi = math.pi

    def forward(self, rho, u, theta):
        return (rho / torch.sqrt(2 * self.pi * theta) ** self.dim) * torch.exp(
            -((u.unsqueeze(-2) - self.v) ** 2).sum(dim=-1) / (2 * theta)
        )

def fsum(f, w):
    return (f * w).sum(dim=-1, keepdim=True)

class _M0(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self.register_buffer('w', w)
    def forward(self, f):
        return fsum(f, self.w)

def _m1_d1(f, v, w):
    return fsum(f * v[..., 0], w)

def _m1_d2(f, v, w):
    return torch.cat([fsum(f * v[..., 0], w), fsum(f * v[..., 1], w)], dim=-1)

def _m1_d3(f, v, w):
    return torch.cat([fsum(f * v[..., 0], w), fsum(f * v[..., 1], w), fsum(f * v[..., 2], w)], dim=-1)

class _M1(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self.register_buffer('v', v)
        self.register_buffer('w', w)
        self.dim = v.shape[-1]
        if self.dim == 1: self._m1 = _m1_d1
        elif self.dim == 2: self._m1 = _m1_d2
        elif self.dim == 3: self._m1 = _m1_d3

    def forward(self, f):
        return self._m1(f, self.v, self.w)

def _m2_d1(f, v, w):
    return fsum(f * v[..., 0] ** 2, w)

def _m2_d2(f, v, w):
    return torch.cat([fsum(f * v[..., 0] ** 2, w), fsum(f * v[..., 1] ** 2, w)], dim=-1)

def _m2_d3(f, v, w):
    return torch.cat([fsum(f * v[..., 0] ** 2, w), fsum(f * v[..., 1] ** 2, w), fsum(f * v[..., 2] ** 2, w)], dim=-1)

class _M2(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self.register_buffer('v', v)
        self.register_buffer('w', w)
        self.dim = v.shape[-1]
        if self.dim == 1: self._m2 = _m2_d1
        elif self.dim == 2: self._m2 = _m2_d2
        elif self.dim == 3: self._m2 = _m2_d3

    def forward(self, f):
        return self._m2(f, self.v, self.w)

class _M012(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self._m0 = _M0(v, w)
        self._m1 = _M1(v, w)
        self._m2 = _M2(v, w)

    def forward(self, f):
        return self._m0(f), self._m1(f), self._m2(f)

class RhoUTheta(nn.Module):
    def __init__(self, v, w, eps=1e-3):
        super().__init__()
        self._m0 = _M0(v, w)
        self._m1 = _M1(v, w)
        self._m2 = _M2(v, w)
        self.eps = eps
        self.dim = v.shape[-1]

    def forward(self, f):
        m0, m1, m2 = self._m0(f), self._m1(f), self._m2(f)
        density = torch.maximum(m0, torch.tensor(self.eps, device=m0.device))
        veloc = m1 / m0
        v2 = (veloc**2).sum(dim=-1, keepdim=True)
        temperature = (m2.sum(dim=-1, keepdim=True) / m0 - v2) / self.dim
        temperature = torch.maximum(temperature, torch.tensor(self.eps, device=temperature.device))
        return density, veloc, temperature

class PrimNorm(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self._m012 = _M012(v, w)
        self.criterion_norm = lambda x: torch.square(x).mean(dim=0)

    def forward(self, f):
        m1, m2, m3 = self._m012(f)
        return torch.cat([self.criterion_norm(m1), self.criterion_norm(m2), self.criterion_norm(m3)], dim=-1)

class MultiResInput(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.register_buffer('freq', torch.tensor(freq, dtype=torch.float32))

    def forward(self, x):
        xf = x[..., None] * self.freq
        return xf.view(xf.shape[:-2] + (xf.shape[-2] * xf.shape[-1],))

class MultiRes(nn.Module):
    def __init__(self, in_channel, out_channel, layers, neurons, freq=(1, 4, 16)):
        super().__init__()
        self.minput = MultiResInput(freq)
        mods = []
        c_in = in_channel * len(freq)
        for i in range(layers):
            mods.append(nn.Linear(c_in if i == 0 else neurons, neurons))
            mods.append(Sine())
        mods.append(nn.Linear(neurons, out_channel))
        self.net = nn.Sequential(*mods)
        
        # Init weights uniformly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                bound = math.sqrt(1 / m.in_features)
                nn.init.uniform_(m.weight, -bound, bound)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(self.minput(x))

class SplitNet(nn.Module):
    def __init__(self, in_channel, layers, neurons, vdis, alpha=0.01):
        super().__init__()
        self.net_eq = MultiRes(in_channel, 5, layers, neurons)
        self.net_neq = MultiRes(in_channel, vdis.shape[0], layers, neurons)
        self.maxwellian = Maxwellian(vdis)
        self.alpha = alpha

    def forward(self, xt):
        www = self.net_eq(xt)
        rho, u, theta = www[..., 0:1], www[..., 1:4], www[..., 4:5]
        rho = torch.exp(-rho)
        theta = torch.exp(-theta)
        x1 = self.maxwellian(rho, u, theta)
        x2 = self.net_neq(xt)
        return x1 * (x1 + self.alpha * x2)

class JacFwd(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        # Using PyTorch functorch for AutoGrad forward Jacobian
        def single_net(x_single):
            return self.net(x_single.unsqueeze(0)).squeeze(0)
        
        jacobian = vmap(jacfwd(single_net))(x) # shape: [Batch, Nv, 2]
        fx = jacobian[..., 0]
        ft = jacobian[..., 1]
        return out, torch.stack((fx, ft))

class MtlLoss(nn.Module):
    def __init__(self, num_losses, eta=1e-3):
        super().__init__()
        self.num_losses = num_losses
        self.params = nn.Parameter(torch.ones(num_losses, dtype=torch.float32))
        self.eta = eta

    def forward(self, losses):
        ww = self.params**2 + self.eta**2
        loss = 0.5 / ww * losses + torch.log(1 + ww)
        return loss.sum() / self.num_losses

# -------- Low Rank (LR) Cells --------
def maxwellian_lr_1d(v, rho, u, theta):
    return rho / torch.sqrt(2 * np.pi * theta) * torch.exp(-((u - v) ** 2) / (2 * theta))

def maxwellian_lr_3d(vtuple, rho, u, theta):
    vx, vy, vz = vtuple
    f1 = maxwellian_lr_1d(vx, rho ** (1 / 3), u[..., 0:1], theta)
    f2 = maxwellian_lr_1d(vy, rho ** (1 / 3), u[..., 1:2], theta)
    f3 = maxwellian_lr_1d(vz, rho ** (1 / 3), u[..., 2:3], theta)
    return f1[..., None], f2[..., None], f3[..., None]

class MaxwellianLR(nn.Module):
    def __init__(self, vtuple):
        super().__init__()
        self.v = vtuple

    def forward(self, rho, u, theta):
        return maxwellian_lr_3d(self.v, rho, u, theta)

def f_sum_lowrank(ft, wt):
    fx, fy, fz = ft
    wx, wy, wz = wt
    sx = (fx * wx.unsqueeze(-1)).sum(dim=-2)
    sy = (fy * wy.unsqueeze(-1)).sum(dim=-2)
    sz = (fz * wz.unsqueeze(-1)).sum(dim=-2)
    return (sx * sy * sz).sum(dim=-1)

def f_m0_lowrank(ft, wt):
    return f_sum_lowrank(ft, wt)[..., None]

def f_m1_lowrank(ft, vt, wt):
    fx, fy, fz = ft
    vx, vy, vz = vt
    mux = f_sum_lowrank((fx * vx[..., None], fy, fz), wt)
    muy = f_sum_lowrank((fx, fy * vy[..., None], fz), wt)
    muz = f_sum_lowrank((fx, fy, fz * vz[..., None]), wt)
    return torch.stack([mux, muy, muz], dim=-1)

def f_m2_lowrank(ft, vt, wt):
    fx, fy, fz = ft
    vx, vy, vz = vt
    mux = f_sum_lowrank((fx * vx[..., None] ** 2, fy, fz), wt)
    muy = f_sum_lowrank((fx, fy * vy[..., None] ** 2, fz), wt)
    muz = f_sum_lowrank((fx, fy, fz * vz[..., None] ** 2), wt)
    return torch.stack([mux, muy, muz], dim=-1)

def f_m012_lowrank(f, v, w):
    return f_m0_lowrank(f, w), f_m1_lowrank(f, v, w), f_m2_lowrank(f, v, w)

class RhoUThetaLr(nn.Module):
    def __init__(self, v, w, eps=1e-3):
        super().__init__()
        self.v, self.w = v, w
        self.eps = eps

    def forward(self, ft):
        m0, m1, m2 = f_m012_lowrank(ft, self.v, self.w)
        density = torch.maximum(m0, torch.tensor(self.eps, device=m0.device))
        veloc = m1 / density
        v2 = (veloc**2).sum(dim=-1, keepdim=True)
        temperature = (m2.sum(dim=-1, keepdim=True) / density - v2) / 3.0
        temperature = torch.maximum(temperature, torch.tensor(self.eps, device=temperature.device))
        return density, veloc, temperature

def rho_u_theta_lowrank(ft, vt, wt):
    eps_r = 1e-4
    m0, m1, m2 = f_m012_lowrank(ft, vt, wt)
    density = torch.maximum(m0, torch.tensor(eps_r, device=m0.device))
    veloc = m1 / density
    v2 = (veloc**2).sum(dim=-1, keepdim=True)
    temperature = (m2.sum(dim=-1, keepdim=True) / density - v2) / 3.0
    return density, veloc, temperature

class PrimNormLR(nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self.v, self.w = v, w
        self.criterion_norm = lambda x: torch.square(x).mean(dim=0)

    def forward(self, f):
        m1, m2, m3 = f_m012_lowrank(f, self.v, self.w)
        return torch.cat([self.criterion_norm(m1), self.criterion_norm(m2), self.criterion_norm(m3)], dim=-1)

def dis_lowrank_add(dislist):
    return (
        torch.cat([d[0] for d in dislist], dim=-1),
        torch.cat([d[1] for d in dislist], dim=-1),
        torch.cat([d[2] for d in dislist], dim=-1),
    )

def dis_lowrank_sub(dis1, dis2):
    return (
        torch.cat([dis1[0], -dis2[0]], dim=-1),
        torch.cat([dis1[1], dis2[1]], dim=-1),
        torch.cat([dis1[2], dis2[2]], dim=-1),
    )

class JacFwdLR(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        def single_net(x_single):
            return self.net(x_single.unsqueeze(0))
        # Requires handling Tuple outputs for LR, simple PyTorch impl
        jacobian = vmap(jacfwd(single_net))(x)
        gx = (jacobian[0][..., 0, 0], jacobian[1][..., 0, 0], jacobian[2][..., 0, 0])
        gt = (jacobian[0][..., 0, 1], jacobian[1][..., 0, 1], jacobian[2][..., 0, 1])
        return out, (gx, gt)

class SplitNetLR(nn.Module):
    def __init__(self, in_channel, layers, neurons, vtuple, rank=40, alpha=0.01):
        super().__init__()
        self.net_eq = MultiRes(in_channel, 5, layers, neurons)
        self.vt = vtuple
        self.rank = rank
        self.out_channel_1, self.out_channel_2, self.out_channel_3 = [v.shape[0] for v in self.vt]
        
        self.net_neq1 = MultiRes(in_channel, self.out_channel_1 * rank, layers, neurons)
        self.net_neq2 = MultiRes(in_channel, self.out_channel_2 * rank, layers, neurons)
        self.net_neq3 = MultiRes(in_channel, self.out_channel_3 * rank, layers, neurons)
        self.maxwellian = MaxwellianLR(self.vt)

    def forward(self, xt):
        www = self.net_eq(xt)
        rho, u, theta = www[..., 0:1], www[..., 1:4], www[..., 4:5]
        rho, theta = torch.exp(-rho), torch.exp(-theta)

        fmx, fmy, fmz = self.maxwellian(rho, u, theta)

        f2x = self.net_neq1(xt).view(-1, self.out_channel_1, self.rank)
        f2x = (0.01**0.33) * fmx * f2x

        f2y = self.net_neq2(xt).view(-1, self.out_channel_2, self.rank)
        f2y = (0.01**0.33) * fmy * f2y

        f2z = self.net_neq3(xt).view(-1, self.out_channel_3, self.rank)
        f2z = (0.01**0.33) * fmz * f2z

        return (torch.cat([fmx, f2x], dim=-1),
                torch.cat([fmy, f2y], dim=-1),
                torch.cat([fmz, f2z], dim=-1))

def lrmse_adap(p, q, r, w1, w2, w3):
    w1, w2, w3 = w1[None, ..., None].sqrt(), w2[None, ..., None].sqrt(), w3[None, ..., None].sqrt()
    p, q, r = p * w1, q * w2, r * w3
    return 0.5 * ((p.mT @ p) * (q.mT @ q) * (r.mT @ r)).sum() * (1.0 / p.shape[0])

class AdaptiveMSE(nn.Module):
    def __init__(self, nx, ny, nz, eta=1e-6):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(nx, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(ny, dtype=torch.float32))
        self.w3 = nn.Parameter(torch.ones(nz, dtype=torch.float32))
        self.eta = eta

    def forward(self, f):
        p, q, r = f
        w1, w2, w3 = self.eta**2 + self.w1**2, self.eta**2 + self.w2**2, self.eta**2 + self.w3**2
        w = w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
        loss_1 = lrmse_adap(p, q, r, 0.5 / w1, 0.5 / w2, 0.5 / w3)
        loss_w = torch.log(1 + w).sum()
        return loss_1, loss_w