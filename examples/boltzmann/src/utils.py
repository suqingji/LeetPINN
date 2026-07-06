# src/utils.py
import math
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from scipy import special
import matplotlib.pyplot as plt
import torch

def fftshift_pt(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = [x.shape[axes] // 2]
        axes = [axes]
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return torch.roll(x, shifts=shift, dims=axes)

def get_gamma(ck):
    return (ck + 5) / (ck + 3)

def get_potential(omega):
    if omega == 0.5:
        alpha = 1.0
    else:
        eta = 4.0 / (2.0 * omega - 1.0) + 1.0
        alpha = (eta - 5.0) / (eta - 1.0)
    return alpha

def get_mu(alpha, omega, kn):
    mu = (5 * (alpha + 1) * (alpha + 2) * math.sqrt(math.pi) /
          (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega)) * kn)
    return mu

def get_kn_bzm(alpha, mu_ref):
    kn_bzm = (64 * math.sqrt(2.0) ** alpha / 5.0 * math.gamma((alpha + 3) / 2) *
              math.gamma(2.0) * math.sqrt(math.pi) * mu_ref)
    return kn_bzm

def lgwt(n: int, a: float, b: float):
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1) * (b - a) + a
    w = w * 0.5 * (b - a)
    return x, w

def init_kernel_mode_vector(vmin, vmax, nv, quad_num: int = 64, omega: float = 0.81, m=5, dtype=np.float64):
    pi = math.pi
    alpha = get_potential(omega)
    umax, vmax, wmax = vmax
    umin, vmin, wmin = vmin
    unum, vnum, wnum = nv
    du, dv, dw = (umax - umin) / (unum - 1), (vmax - vmin) / (vnum - 1), (wmax - wmin) / (wnum - 1)
    supp = math.sqrt(2.0) * 2.0 * max(umax, vmax, wmax) / (3.0 + math.sqrt(2.0))

    fre_vx = np.linspace(-pi / du, (unum / 2 - 1.0) * 2.0 * pi / unum / du, unum)
    fre_vy = np.linspace(-pi / dv, (vnum / 2 - 1.0) * 2.0 * pi / vnum / dv, vnum)
    fre_vz = np.linspace(-pi / dw, (wnum / 2 - 1.0) * 2.0 * pi / wnum / dw, wnum)

    abscissa, gweight = lgwt(quad_num, 0.0, supp)
    theta = pi / m * np.arange(1, m - 1 + 1)
    theta2 = pi / m * np.arange(1, m + 1)

    s = ((fre_vx[:, None, None] * np.sin(theta)[None, :, None] * np.cos(theta2)[None, None, :])[:, None, None, :, :] +
         (fre_vy[:, None, None] * np.sin(theta)[None, :, None] * np.sin(theta2)[None, None, :])[None, :, None, :, :] +
         (fre_vz[:, None, None] * np.cos(theta)[None, :, None])[None, None, :, :, :])

    int_temp = (2 * gweight[..., None, None, None, None, None] *
                np.cos(s[None, ...] * abscissa[..., None, None, None, None, None]) *
                (abscissa[..., None, None, None, None, None] ** alpha)).sum(axis=0)
    phi2 = int_temp * np.sin(theta[None, None, None, :, None])

    s = ((fre_vx * fre_vx)[:, None, None, None, None] +
         (fre_vy * fre_vy)[None, :, None, None, None] +
         (fre_vz * fre_vz)[None, None, :, None, None] - s * s)

    so = s.copy()
    s = np.abs(s)
    s = np.sqrt(s)
    bel = supp * s
    bessel = special.jv(1, bel)

    psi2 = pi * supp * supp * np.ones_like(s)
    np.divide(2.0 * pi * supp * bessel, s, out=psi2, where=so > 0)

    phipsi2 = (phi2 * psi2).sum(axis=(-1, -2))
    return phi2, psi2, phipsi2

def collision_fft_fg(f_spec, g_spec, kn_bzm, phi, psi, phipsi):
    unum, vnum, wnum = phi.shape[:3]
    f_spec = ifftn(f_spec, axes=(-3, -2, -1), norm="forward")
    g_spec = ifftn(g_spec, axes=(-3, -2, -1), norm="forward")
    f_spec = f_spec / (unum * vnum * wnum)
    g_spec = g_spec / (unum * vnum * wnum)

    f_spec = fftshift(f_spec, axes=(-3, -2, -1))
    g_spec = fftshift(g_spec, axes=(-3, -2, -1))
    f_temp = 0
    m = phi.shape[-1]
    for i in range(1, m - 1 + 1):
        for j in range(1, m + 1):
            fc1 = f_spec * phi[:, :, :, i - 1, j - 1]
            fc2 = g_spec * psi[:, :, :, i - 1, j - 1]
            fc11 = fftn(fc1, axes=(-3, -2, -1), norm="backward")
            fc22 = fftn(fc2, axes=(-3, -2, -1), norm="backward")
            f_temp = f_temp + fc11 * fc22
    fc1 = f_spec * phipsi
    fc2 = g_spec
    fc11 = fftn(fc1, axes=(-3, -2, -1), norm="backward")
    fc22 = fftn(fc2, axes=(-3, -2, -1), norm="backward")
    f_temp = f_temp - fc11 * fc22
    q = 4.0 * np.pi**2 / kn_bzm / m**2 * f_temp.real
    return q

def orthonormalize(vectors: np.array) -> np.array:
    assert vectors.shape[1] <= vectors.shape[0], "number of vectors must be <= dimension"
    orthonormalized_vectors = np.zeros_like(vectors)
    orthonormalized_vectors[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=0, ord=2)
    for i in range(1, orthonormalized_vectors.shape[1]):
        vector = vectors[:, i]
        v = orthonormalized_vectors[:, :i]
        pv_vector = v @ (v.T @ vector)
        orthonormalized_vectors[:, i] = (vector - pv_vector) / np.linalg.norm(vector - pv_vector, axis=0, ord=2)
    return orthonormalized_vectors

def fvmlinspace(vmin, vmax, nv):
    dv = (vmax - vmin) / nv
    return np.linspace(vmin + dv / 2, vmax - dv / 2, nv)

def vmsh(vmin, vmax, nv):
    v = fvmlinspace(vmin, vmax, nv)
    w = (vmax - vmin) / nv
    return v, w

def mesh_nd(vmin, vmax, nv, return_list=False):
    if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and isinstance(nv, int):
        v, w = vmsh(vmin, vmax, nv)
        vlist, wlist = [v,], [w,]
    else:
        vlist, wlist = list(zip(*[vmsh(vmini, vmaxi, nvi) for vmini, vmaxi, nvi in zip(vmin, vmax, nv)]))
        v = np.meshgrid(*vlist, indexing="ij")
        v = np.stack([vi.flatten() for vi in v], axis=-1)
        w = np.multiply.reduce(wlist)
        wlist = [w * np.ones_like(v) for v, w in zip(vlist, wlist)]
    return (vlist, wlist) if return_list else (v, w)

def get_mesh(vconfig, return_list=False):
    nv, vmin, vmax = vconfig["nv"], vconfig["vmin"], vconfig["vmax"]
    return mesh_nd(vmin, vmax, nv, return_list)

def get_vdis(vconfig):
    v, w = get_mesh(vconfig)
    return torch.tensor(v, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)

def get_vtuple(vconfig):
    vlist, wlist = get_mesh(vconfig, return_list=True)
    vtuple = tuple([torch.tensor(v, dtype=torch.float32) for v in vlist])
    wtuple = tuple([torch.tensor(w, dtype=torch.float32) for w in wlist])
    return vtuple, wtuple

def get_new_kernel(f_bases, f_bases2, nx, ny, nz, kn_bzm, phi, psi, phipsi):
    k = f_bases.shape[1]
    t = np.zeros((k, k, k))
    for i in range(k):
        for j in range(k):
            f1 = f_bases[:, i].reshape((nx, ny, nz))
            f2 = f_bases[:, j].reshape((nx, ny, nz))
            f3 = collision_fft_fg(f1, f2, kn_bzm, phi, psi, phipsi).reshape((nx * ny * nz, 1))
            coef = f_bases2.T @ f3
            t[i, j, :] = coef[..., 0]
    return t

def visual(problem, resolution=100, filename="result.jpg"):
    device = next(problem.parameters()).device
    x = np.linspace(-0.5, 0.5, resolution)
    t0 = 0.0 * np.ones_like(x)
    t1 = 0.1 * np.ones_like(x)
    xt0 = torch.tensor(np.stack((x, t0), axis=-1), dtype=torch.float32, device=device)
    xt1 = torch.tensor(np.stack((x, t1), axis=-1), dtype=torch.float32, device=device)
    problem.eval()
    with torch.no_grad():
        rho0, u0, theta0 = problem.pred(xt0)
        rho1, u1, theta1 = problem.pred(xt1)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, rho0.cpu().numpy(), label=r"$\rho$")
    ax[0].plot(x, u0[..., 0].cpu().numpy(), label="$u_x$")
    ax[0].plot(x, theta0.cpu().numpy(), label="T")
    ax[0].legend()
    ax[1].plot(x, rho1.cpu().numpy(), label=r"$\rho$")
    ax[1].plot(x, u1[..., 0].cpu().numpy(), label="$u_x$")
    ax[1].plot(x, theta1.cpu().numpy(), label="T")
    ax[1].legend()
    fig.savefig(filename)
    problem.train()
    return fig

def valid_model(config, problem):
    device = next(problem.parameters()).device
    ref_solution0 = np.load(config["ref_solution"])
    rho0_ref, u0_ref, theta0_ref = ref_solution0["rho0"], ref_solution0["u0"], ref_solution0["T0"]
    rho1_ref, u1_ref, theta1_ref = ref_solution0["rho1"], ref_solution0["u1"], ref_solution0["T1"]
    resolution = rho0_ref.shape[0]
    x = np.linspace(-0.5, 0.5, resolution)
    t0 = 0.0 * np.ones_like(x)
    t1 = 0.1 * np.ones_like(x)
    xt0 = torch.tensor(np.stack((x, t0), axis=-1), dtype=torch.float32, device=device)
    xt1 = torch.tensor(np.stack((x, t1), axis=-1), dtype=torch.float32, device=device)
    
    problem.eval()
    with torch.no_grad():
        rho0, u0, theta0 = problem.pred(xt0)
        rho1, u1, theta1 = problem.pred(xt1)
    problem.train()

    err1 = (((rho0.cpu().numpy()[..., 0] - rho0_ref) ** 2).mean() / ((rho0_ref) ** 2).mean()) ** 0.5
    err2 = (((u0.cpu().numpy()[..., 0] - u0_ref) ** 2).mean() / (1 + (u0_ref) ** 2).mean()) ** 0.5
    err3 = (((theta0.cpu().numpy()[..., 0] - theta0_ref) ** 2).mean() / (theta0_ref**2).mean()) ** 0.5
    print(f"err at t=0.0: {err1:.3e}\t{err2:.3e}\t{err3:.3e}\t")

    err1 = (((rho1.cpu().numpy()[..., 0] - rho1_ref) ** 2).mean() / ((rho1_ref) ** 2).mean()) ** 0.5
    err2 = (((u1.cpu().numpy()[..., 0] - u1_ref) ** 2).mean() / (1 + (u1_ref) ** 2).mean()) ** 0.5
    err3 = (((theta1.cpu().numpy()[..., 0] - theta1_ref) ** 2).mean() / (theta1_ref**2).mean()) ** 0.5
    print(f"err at t=0.1: {err1:.3e}\t{err2:.3e}\t{err3:.3e}\t")

def save_points(problem, points=1000, filename="points.npz"):
    device = next(problem.parameters()).device
    x = np.random.rand(points) - 0.5
    t = 0.1 * np.random.rand(points)
    xt0 = torch.tensor(np.stack((x, t), axis=-1), dtype=torch.float32, device=device)
    problem.eval()
    with torch.no_grad():
        f = problem.net(xt0)
    np.savez(filename, f=f.cpu().numpy())
    problem.train()