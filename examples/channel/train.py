"""PyTorch PINN for steady 2D channel flow.

The example solves the nondimensional Poiseuille flow in a straight channel:
    u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) = 0
    u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) = 0
    u_x + v_y = 0

Domain: x in [0, L], y in [-1, 1].
Analytic solution with outlet pressure p(L, y) = 0:
    u = umax * (1 - y^2), v = 0, p = 2 * nu * umax * (L - x).
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
from torch import nn


torch.manual_seed(123456)


@dataclass
class ChannelConfig:
    length: float
    nu: float
    umax: float
    n_f: int
    n_inlet: int
    n_wall: int
    n_outlet: int


class MLP(nn.Module):
    """Fully connected network with tanh activations."""

    def __init__(self, in_dim=2, out_dim=3, hidden_dim=64, num_layers=5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, xy):
        return self.net(xy)


def parse_args():
    parser = argparse.ArgumentParser(description="PINN channel flow solver in PyTorch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--umax", type=float, default=1.0)
    parser.add_argument("--n_f", type=int, default=4096, help="Interior collocation points")
    parser.add_argument("--n_inlet", type=int, default=512)
    parser.add_argument("--n_wall", type=int, default=1024)
    parser.add_argument("--n_outlet", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./outputs/channel")
    parser.add_argument("--use_lbfgs", action="store_true", help="Run a short L-BFGS refinement after Adam")
    return parser.parse_args()


def sample_training_points(config, device):
    x_f = config.length * torch.rand(config.n_f, 1, device=device)
    y_f = 2.0 * torch.rand(config.n_f, 1, device=device) - 1.0
    xy_f = torch.cat([x_f, y_f], dim=1).requires_grad_(True)

    y_in = torch.linspace(-1.0, 1.0, config.n_inlet, device=device).view(-1, 1)
    xy_in = torch.cat([torch.zeros_like(y_in), y_in], dim=1)

    x_wall = config.length * torch.rand(config.n_wall, 1, device=device)
    y_bottom = -torch.ones(config.n_wall // 2, 1, device=device)
    y_top = torch.ones(config.n_wall - config.n_wall // 2, 1, device=device)
    xy_bottom = torch.cat([x_wall[: y_bottom.shape[0]], y_bottom], dim=1)
    xy_top = torch.cat([x_wall[y_bottom.shape[0] :], y_top], dim=1)
    xy_wall = torch.cat([xy_bottom, xy_top], dim=0)

    y_out = torch.linspace(-1.0, 1.0, config.n_outlet, device=device).view(-1, 1)
    xy_out = torch.cat([config.length * torch.ones_like(y_out), y_out], dim=1)
    return xy_f, xy_in, xy_wall, xy_out


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def analytic_solution(xy, config):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    u = config.umax * (1.0 - y**2)
    v = torch.zeros_like(u)
    p = 2.0 * config.nu * config.umax * (config.length - x)
    return u, v, p


def pde_residual(model, xy, config):
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du = grad(u, xy)
    dv = grad(v, xy)
    dp = grad(p, xy)
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]

    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]

    momentum_x = u * u_x + v * u_y + p_x - config.nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - config.nu * (v_xx + v_yy)
    continuity = u_x + v_y
    return momentum_x, momentum_y, continuity


def pinn_loss(model, points, config):
    xy_f, xy_in, xy_wall, xy_out = points
    mse = nn.functional.mse_loss

    mom_x, mom_y, cont = pde_residual(model, xy_f, config)
    zero_f = torch.zeros_like(mom_x)
    loss_pde = mse(mom_x, zero_f) + mse(mom_y, zero_f) + mse(cont, zero_f)

    inlet_pred = model(xy_in)
    u_in, v_in, _ = analytic_solution(xy_in, config)
    loss_inlet = mse(inlet_pred[:, 0:1], u_in) + mse(inlet_pred[:, 1:2], v_in)

    wall_pred = model(xy_wall)
    loss_wall = mse(wall_pred[:, 0:1], torch.zeros_like(wall_pred[:, 0:1]))
    loss_wall = loss_wall + mse(wall_pred[:, 1:2], torch.zeros_like(wall_pred[:, 1:2]))

    outlet_pred = model(xy_out)
    loss_outlet = mse(outlet_pred[:, 2:3], torch.zeros_like(outlet_pred[:, 2:3]))

    total = loss_pde + 10.0 * loss_inlet + 10.0 * loss_wall + loss_outlet
    parts = {
        "pde": loss_pde.detach(),
        "inlet": loss_inlet.detach(),
        "wall": loss_wall.detach(),
        "outlet": loss_outlet.detach(),
    }
    return total, parts


@torch.no_grad()
def evaluate(model, config, device, resolution=101):
    x = torch.linspace(0.0, config.length, resolution, device=device)
    y = torch.linspace(-1.0, 1.0, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    pred = model(xy)
    truth = torch.cat(analytic_solution(xy, config), dim=1)
    rel_l2 = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    return rel_l2.item(), xy.detach().cpu(), pred.detach().cpu(), truth.detach().cpu()


def save_plots(history, xy, pred, truth, config, save_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plotting.")
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.semilogy(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=160)
    plt.close()

    n = int(math.sqrt(xy.shape[0]))
    x_grid = xy[:, 0].reshape(n, n)
    y_grid = xy[:, 1].reshape(n, n)
    u_pred = pred[:, 0].reshape(n, n)
    u_true = truth[:, 0].reshape(n, n)
    u_error = torch.abs(u_pred - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        [u_pred.numpy(), u_true.numpy(), u_error.numpy()],
        ["PINN u", "Analytic u", "|error|"],
    ):
        im = ax.contourf(x_grid.numpy(), y_grid.numpy(), field, levels=40, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(save_dir, "velocity_u.png"), dpi=160)
    plt.close(fig)


def train(args):
    device = torch.device(args.device)
    config = ChannelConfig(
        length=args.length,
        nu=args.nu,
        umax=args.umax,
        n_f=args.n_f,
        n_inlet=args.n_inlet,
        n_wall=args.n_wall,
        n_outlet=args.n_outlet,
    )
    points = sample_training_points(config, device)
    model = MLP(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Running on {device.type.upper()}.")
    print(f"Training points: interior={args.n_f}, inlet={args.n_inlet}, wall={args.n_wall}, outlet={args.n_outlet}")

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss, parts = pinn_loss(model, points, config)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if epoch == 1 or epoch % args.print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"epoch: {epoch:6d} loss: {loss.item():.6e} "
                f"pde: {parts['pde'].item():.2e} inlet: {parts['inlet'].item():.2e} "
                f"wall: {parts['wall'].item():.2e} outlet: {parts['outlet'].item():.2e} "
                f"time: {elapsed:.1f}s"
            )

    if args.use_lbfgs:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=500, history_size=50)

        def closure():
            lbfgs.zero_grad()
            loss, _ = pinn_loss(model, points, config)
            loss.backward()
            return loss

        print("Running L-BFGS refinement...")
        lbfgs.step(closure)

    rel_l2, xy, pred, truth = evaluate(model, config, device)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))
    save_plots(history, xy, pred, truth, config, args.save_dir)
    print(f"Relative L2 error against analytic solution: {rel_l2:.6e}")
    print(f"Saved results to: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    train(parse_args())
