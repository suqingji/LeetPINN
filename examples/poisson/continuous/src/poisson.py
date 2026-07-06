import math
import torch
import torch.nn as nn

class MTLWeightedLoss(nn.Module):
    def __init__(self, num_losses=2):
        super(MTLWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum

class PoissonPINN(nn.Module):
    """Define the loss of the Poisson equation."""
    def __init__(self, model, n_dim):
        super(PoissonPINN, self).__init__()
        self.model = model
        self.n_dim = n_dim
        self.loss_fn = MTLWeightedLoss(num_losses=2)

    def pde_loss(self, x_domain):
        """Define the governing equation loss."""
        x_domain.requires_grad_(True)
        u = self.model(x_domain)

        laplacian = 0.0
        # 一阶导数
        u_x = torch.autograd.grad(u.sum(), x_domain, create_graph=True)[0]
        # 二阶导数 (拉普拉斯算子)
        for i in range(self.n_dim):
            u_xx = torch.autograd.grad(u_x[:, i].sum(), x_domain, create_graph=True)[0][:, i]
            laplacian += u_xx

        # 源项: prod(sin(4 * pi * x_i))
        src_term = torch.prod(torch.sin(4 * math.pi * x_domain), dim=1)
        pde_res = laplacian + src_term
        return torch.mean(pde_res ** 2)

    def bc_loss(self, x_bc):
        """Define the Dirichlet boundary condition loss."""
        u = self.model(x_bc)
        bc_term = torch.prod(torch.sin(4 * math.pi * x_bc), dim=1, keepdim=True)
        bc_term = bc_term / (16.0 * self.n_dim * math.pi * math.pi)
        
        bc_res = u - bc_term
        return torch.mean(bc_res ** 2)

    def forward(self, pde_data, bc_data):
        loss_pde = self.pde_loss(pde_data)
        loss_bc = self.bc_loss(bc_data)
        return self.loss_fn([loss_pde, loss_bc]), loss_pde, loss_bc