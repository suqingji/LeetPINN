# src/boltzmann.py
import numpy as np
import torch
import torch.nn as nn
from src.utils import init_kernel_mode_vector, get_vdis, get_vtuple, fftshift_pt, get_new_kernel
from src.cells import (
    Maxwellian, RhoUTheta, MtlLoss, JacFwd, PrimNorm, dis_lowrank_add, dis_lowrank_sub, 
    JacFwdLR, MaxwellianLR, RhoUThetaLr, PrimNormLR, rho_u_theta_lowrank, AdaptiveMSE
)

class BGKKernel(nn.Module):
    def __init__(self, vmin, vmax, nv):
        super().__init__()
        v, w = get_vdis({"vmin": vmin, "vmax": vmax, "nv": nv})
        self.register_buffer('vdis', v)
        self.register_buffer('wdis', w)
        self.maxwellian_nd = Maxwellian(self.vdis)
        self.rho_u_theta = RhoUTheta(self.vdis, self.wdis)

    def forward(self, f, kn=1):
        rho, u, theta = self.rho_u_theta(f)
        f_m = self.maxwellian_nd(rho, u, theta)
        return 1 / kn * (f_m - f)

class FBGKKernel(nn.Module):
    def __init__(self, vmin, vmax, nv, omega):
        super().__init__()
        v, w = get_vdis({"vmin": vmin, "vmax": vmax, "nv": nv})
        self.register_buffer('vdis', v)
        self.register_buffer('wdis', w)
        self.maxwellian_nd = Maxwellian(self.vdis)
        self.rho_u_theta = RhoUTheta(self.vdis, self.wdis)
        self.omega = omega

    def forward(self, f, mu_ref=1):
        rho, u, theta = self.rho_u_theta(f)
        f_m = self.maxwellian_nd(rho, u, theta)
        tau = mu_ref * 2 / (theta * 2) ** (1 - self.omega) / rho
        return 1 / tau * (f_m - f)

class FSMKernel(nn.Module):
    def __init__(self, vmin, vmax, nv, quad_num=64, omega=0.81, m=5):
        super().__init__()
        phi, psi, phipsi = init_kernel_mode_vector(vmin, vmax, nv, quad_num, omega, m, np.float32)
        self.register_buffer('phi', torch.tensor(phi, dtype=torch.float32))
        self.register_buffer('psi', torch.tensor(psi, dtype=torch.float32))
        self.register_buffer('phipsi', torch.tensor(phipsi, dtype=torch.float32))
        self.num = nv[0] * nv[1] * nv[2]
        self.nv = (nv[0], nv[1], nv[2])
        self.m = m

    def forward(self, f_ms, kn_bzm=1):
        f_ms = f_ms.view(f_ms.shape[0], *self.nv)
        f_spec = f_ms.to(torch.complex64)
        
        f_spec = torch.fft.ifftn(f_spec, dim=(-3, -2, -1), norm="forward")
        f_spec = f_spec / self.num
        f_spec = fftshift_pt(f_spec, axes=(-3, -2, -1))
        
        f_temp = 0.0
        for i in range(1, self.m):
            for j in range(1, self.m + 1):
                fc1 = f_spec * self.phi[:, :, :, i - 1, j - 1]
                fc2 = f_spec * self.psi[:, :, :, i - 1, j - 1]
                fc11 = torch.fft.fftn(fc1, dim=(-3, -2, -1), norm="backward")
                fc22 = torch.fft.fftn(fc2, dim=(-3, -2, -1), norm="backward")
                f_temp = f_temp + fc11 * fc22

        fc1 = f_spec * self.phipsi
        fc2 = f_spec
        fc11 = torch.fft.fftn(fc1, dim=(-3, -2, -1), norm="backward")
        fc22 = torch.fft.fftn(fc2, dim=(-3, -2, -1), norm="backward")
        f_temp = f_temp - fc11 * fc22
        
        q = 4.0 * np.pi**2 / kn_bzm / self.m**2 * f_temp.real
        return q.view(-1, self.num)

class FSMLAKernel(nn.Module):
    def __init__(self, f, k, g):
        super().__init__()
        self.register_buffer('f', f)
        self.register_buffer('k', k)
        self.register_buffer('g', g)

    def forward(self, x, kn_bzm=1.0):
        ff = x @ self.f
        tmp = ff.unsqueeze(-1) * ff.unsqueeze(-2)
        q = (tmp.unsqueeze(-1) * self.k).sum(dim=-2).sum(dim=-2)
        qr = q @ self.g.T
        return qr / kn_bzm

class BoltzmannEqu(nn.Module):
    def __init__(self, net, kn, vconfig, iv_weight=100, bv_weight=100, pde_weight=10):
        super().__init__()
        self.net = net
        self.kn = kn
        vdis, wdis = get_vdis(vconfig)
        self.register_buffer('vdis', vdis)
        
        loss_num = 3 * (vdis.shape[0] + 1 + 2 * vdis.shape[-1])
        self.mtl = MtlLoss(loss_num)
        self.jac = JacFwd(self.net)
        self.iv_weight = iv_weight
        self.bv_weight = bv_weight
        self.pde_weight = pde_weight
        self.maxwellian_nd = Maxwellian(vdis)
        self.rho_u_theta = RhoUTheta(vdis, wdis)
        self.criterion_norm = lambda x: torch.square(x).mean(dim=0)
        self.prim_norm = PrimNorm(vdis, wdis)
        self.collision = BGKKernel(vconfig["vmin"], vconfig["vmax"], vconfig["nv"])

    def pred(self, xt):
        f = self.net(xt)
        return self.rho_u_theta(f)

    def governing_equation(self, inputs):
        f, fxft = self.jac(inputs)
        fx, ft = fxft[0], fxft[1]
        pde = ft + self.vdis[..., 0] * fx - self.collision(f, self.kn)
        return pde

    def boundary_condition(self, bv_points1, bv_points2):
        return self.net(bv_points1) - self.net(bv_points2)

    def initial_condition(self, inputs):
        iv_pred = self.net(inputs)
        iv_x = inputs[..., 0:1]
        rho_l = torch.sin(2 * np.pi * iv_x) * 0.5 + 1
        u_l = torch.zeros((iv_x.shape[0], 3), dtype=torch.float32, device=inputs.device)
        theta_l = torch.sin(2 * np.pi * iv_x + 0.2) * 0.5 + 1
        iv_truth = self.maxwellian_nd(rho_l, u_l, theta_l)
        return iv_pred - iv_truth

    def forward(self, domain_points, iv_points, bv_points1, bv_points2):
        pde = self.governing_equation(domain_points)
        iv = self.initial_condition(iv_points)
        bv = self.boundary_condition(bv_points1, bv_points2)

        loss_pde = self.pde_weight * self.criterion_norm(pde)
        loss_pde2 = self.pde_weight * self.prim_norm(pde)
        loss_bv = self.bv_weight * self.criterion_norm(bv)
        loss_bv2 = self.bv_weight * self.prim_norm(bv)
        loss_iv = self.iv_weight * self.criterion_norm(iv)
        loss_iv2 = self.iv_weight * self.prim_norm(iv)

        loss_sum = self.mtl(torch.cat([loss_iv, loss_iv2, loss_bv, loss_bv2, loss_pde, loss_pde2], dim=-1))
        return loss_sum, (loss_iv, loss_iv2, loss_bv, loss_bv2, loss_pde, loss_pde2)

class BoltzmannBGK(BoltzmannEqu):
    pass 

class BoltzmannFBGK(BoltzmannEqu):
    def __init__(self, net, kn, vconfig, omega=0.81, **kwargs):
        super().__init__(net, kn, vconfig, **kwargs)
        self.collision = FBGKKernel(vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=omega)

class BoltzmannFSM(BoltzmannEqu):
    def __init__(self, net, kn, vconfig, omega=0.81, **kwargs):
        super().__init__(net, kn, vconfig, **kwargs)
        self.collision = FSMKernel(vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=omega)

class BoltzmannLA(BoltzmannEqu):
    def __init__(self, net, kn, vconfig, f, k, g, omega=0.81, **kwargs):
        super().__init__(net, kn, vconfig, **kwargs)
        self.collision = FSMLAKernel(f, k, g)

class BoltzmannLR(nn.Module):
    def __init__(self, net, kn, vconfig, iv_weight=100, bv_weight=100, pde_weight=10):
        super().__init__()
        self.net = net
        self.kn = kn
        vdis, _ = get_vdis(vconfig)
        vtuple, wtuple = get_vtuple(vconfig)
        self.register_buffer('vdis', vdis)
        
        self.vtuple = vtuple
        self.wtuple = wtuple
        self.iv_weight, self.bv_weight, self.pde_weight = iv_weight, bv_weight, pde_weight
        self.nvx, self.nvy, self.nvz = [v.shape[0] for v in vtuple]

        self.adaptive_loss_pde = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_iv = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_bv = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_prim = MtlLoss(3 * 7)

        self.jac = JacFwdLR(self.net)
        self.maxwellian = MaxwellianLR(vtuple)
        self.rho_u_theta = RhoUThetaLr(self.vtuple, self.wtuple)
        self.primnorm = PrimNormLR(vtuple, wtuple)
        self.knr3 = kn ** (1 / 3)

    def pred(self, xt):
        return self.rho_u_theta(self.net(xt))

    def governing_equation(self, inputs):
        f, fxft = self.jac(inputs)
        p, q, r = f
        vx = self.vtuple[0]
        p_x, q_x, r_x = fxft[0]
        p_t, q_t, r_t = fxft[1]
        
        f_t = ((p_t, q, r), (p, q_t, r), (p, q, r_t))
        vf_x = ((p_x * vx[..., None], q, r), (p * vx[..., None], q_x, r), (p * vx[..., None], q, r_x))
        rho, u, theta = rho_u_theta_lowrank(f, self.vtuple, self.wtuple)
        f_mx, f_my, f_mz = self.maxwellian(rho, u, theta)

        f_m = ((-1/self.knr3 * f_mx, -1/self.knr3 * f_my, -1/self.knr3 * f_mz),)
        f_c = ((1/self.knr3 * p, 1/self.knr3 * q, 1/self.knr3 * r),)
        return dis_lowrank_add(f_t + vf_x + f_m + f_c)

    def boundary_condition(self, bv_points1, bv_points2):
        return dis_lowrank_sub(self.net(bv_points1), self.net(bv_points2))

    def initial_condition(self, inputs):
        iv_pred = self.net(inputs)
        iv_x = inputs[..., 0:1]
        rho_l = torch.sin(2 * np.pi * iv_x) * 0.5 + 1
        u_l = torch.zeros((iv_x.shape[0], 3), dtype=torch.float32, device=inputs.device)
        theta_l = torch.sin(2 * np.pi * iv_x + 0.2) * 0.5 + 1
        iv_truth = self.maxwellian(rho_l, u_l, theta_l)
        return dis_lowrank_sub(iv_pred, iv_truth)

    def forward(self, domain_points, iv_points, bv_points1, bv_points2):
        pde = self.governing_equation(domain_points)
        iv = self.initial_condition(iv_points)
        bv = self.boundary_condition(bv_points1, bv_points2)

        loss_bv, loss_bv_w = self.adaptive_loss_bv((self.bv_weight * bv[0], bv[1], bv[2]))
        loss_iv, loss_iv_w = self.adaptive_loss_iv((self.iv_weight * iv[0], iv[1], iv[2]))
        loss_pde, loss_pde_w = self.adaptive_loss_pde((self.pde_weight * pde[0], pde[1], pde[2]))

        loss_bv_p = self.primnorm((self.bv_weight * bv[0], bv[1], bv[2]))
        loss_iv_p = self.primnorm((self.iv_weight * iv[0], iv[1], iv[2]))
        loss_pde_p = self.primnorm((self.pde_weight * pde[0], pde[1], pde[2]))

        loss_sum = (self.adaptive_loss_prim(torch.cat([loss_bv_p, loss_iv_p, loss_pde_p])) +
                    loss_bv + loss_bv_w + loss_iv + loss_iv_w + loss_pde + loss_pde_w)
        return loss_sum, None

def get_reduced_kernel(config, traindata):
    vconfig, rank = config["vmesh"], config["rank"]
    collision = FSMKernel(vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=config["omega"])
    train_tensor = torch.tensor(traindata, dtype=torch.float32)
    q_data = collision(train_tensor, 1.0)

    U, S, Vh = torch.linalg.svd(train_tensor, full_matrices=False)
    U2, S2, Vh2 = torch.linalg.svd(q_data, full_matrices=False)
    
    s, s_2 = S.numpy(), S2.numpy()
    vh, vh_2 = Vh.numpy(), Vh2.numpy()

    vdis, _ = get_vdis(vconfig)
    nvprod = vdis.shape[0]
    
    c_rho = torch.ones((1, nvprod))
    c_veloc = vdis.mT
    c_energy = (vdis.mT**2).sum(dim=0, keepdim=True)
    c_feature = torch.cat([c_rho, c_veloc, c_energy])
    cc_feat = c_feature.numpy().T

    vhs = vh * s[:, None]
    cc = vhs.T
    cc = np.concatenate([cc_feat.T, cc])
    vhsc = orthonormalize(cc.T).T
    f_bases = vhsc[:rank].T

    vh2s = vh_2 * s_2[:, None]
    cc2 = vh2s[:, :rank]
    cc2 = cc2 - cc_feat @ (cc_feat.T @ cc2)
    
    import scipy.linalg
    vh2sc = scipy.linalg.orth(cc2, rcond=1e-10)
    
    nv, vmin, vmax = vconfig["nv"], vconfig["vmin"], vconfig["vmax"]
    phi, psi, phipsi = init_kernel_mode_vector(vmin, vmax, nv, 5)
    nk = get_new_kernel(f_bases, vh2sc, nv[0], nv[1], nv[2], 1.0, phi, psi, phipsi)
    return f_bases, vh2sc, nk