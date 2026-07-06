import math
import torch

def dirichlet(model, x, n_dim):
    """
    Dirichlet 边界: a * u(x) = b
    返回残差: u(x) - bc_term
    """
    u = model(x)
    bc_term = torch.prod(torch.sin(4 * math.pi * x), dim=1, keepdim=True)
    bc_term = bc_term / (16.0 * n_dim * math.pi * math.pi)
    return u - bc_term

def robin(model, x, n_dim):
    """
    Robin 边界: a * u(x) + b * u'(x) = c
    返回残差: u(x) + u_x - bc_term - bc_term_u
    """
    x.requires_grad_(True)
    u = model(x)
    
    # 求解关于 x 的一阶偏导数总和
    grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_x = torch.sum(grads, dim=1, keepdim=True)
    
    # 理论函数的 bc_term
    bc_term = torch.prod(torch.sin(4 * math.pi * x), dim=1, keepdim=True) / (16.0 * n_dim * math.pi * math.pi)
    
    # 理论函数一阶导数的 bc_term_u
    bc_term_u = torch.zeros((x.shape[0], 1), device=x.device)
    for i in range(n_dim):
        term = torch.ones((x.shape[0], 1), device=x.device)
        for j in range(n_dim):
            if i != j:
                term = term * torch.sin(4 * math.pi * x[:, j:j+1])
            else:
                term = term * torch.cos(4 * math.pi * x[:, j:j+1])
        bc_term_u += term
    bc_term_u = bc_term_u / (4.0 * n_dim * math.pi)
    
    return u + u_x - bc_term - bc_term_u

def periodic(model, x, n_dim):
    """
    Periodic 边界 (完全映射原版逻辑)
    返回边界常数项残差
    """
    bc_term = (2 ** n_dim) / (16.0 * n_dim * math.pi * math.pi)
    return torch.full((x.shape[0], 1), bc_term, device=x.device, requires_grad=True)

def get_bc(bc_name):
    """根据配置文件返回对应的边界条件函数"""
    bc_type = {
        "dirichlet": dirichlet,
        "robin": robin,
        "periodic": periodic,
    }
    return bc_type.get(bc_name, dirichlet)