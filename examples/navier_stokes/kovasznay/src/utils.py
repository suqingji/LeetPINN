import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

def relative_l2(x, y):
    """Calculate the relative L2 error."""
    return np.sqrt(np.mean(np.square(x - y))) / np.sqrt(np.mean(np.square(y)))

def calculate_l2_error(problem, model, ds_test, device):
    """Calculate the relative L2 error on test dataset."""
    model.eval()
    with torch.no_grad():
        for x_domain, x_bc in ds_test:
            x_domain, x_bc = x_domain.to(device), x_bc.to(device)
            
            y_pred_domain = model(x_domain)
            y_pred_bc = model(x_bc)

            # Domain 解析解
            domain_true_u = problem.u_func(x_domain[:, 0:1], x_domain[:, 1:2])
            domain_true_v = problem.v_func(x_domain[:, 0:1], x_domain[:, 1:2])
            domain_true_p = problem.p_func(x_domain[:, 0:1], x_domain[:, 1:2])
            domain_true = torch.cat([domain_true_u, domain_true_v, domain_true_p], dim=1)
            
            # Boundary 解析解
            bc_true_u = problem.u_func(x_bc[:, 0:1], x_bc[:, 1:2])
            bc_true_v = problem.v_func(x_bc[:, 0:1], x_bc[:, 1:2])
            bc_true_p = problem.p_func(x_bc[:, 0:1], x_bc[:, 1:2])
            bc_true = torch.cat([bc_true_u, bc_true_v, bc_true_p], dim=1)

            metric_domain = relative_l2(y_pred_domain.cpu().numpy(), domain_true.cpu().numpy())
            metric_bc = relative_l2(y_pred_bc.cpu().numpy(), bc_true.cpu().numpy())
            
            print(f"Relative L2 error on domain: {metric_domain}")
            print(f"Relative L2 error on boundary: {metric_bc}")

def visual(model, config, device, resolution=100):
    """Visualization of the results."""
    model.eval()
    x_flat = np.linspace(0, 1, resolution)
    y_flat = np.linspace(0, 1, resolution)
    y_grid, x_grid = np.meshgrid(x_flat, y_flat)
    
    x = x_grid.reshape((-1, 1))
    y = y_grid.reshape((-1, 1))
    xy = np.concatenate((x, y), axis=1)
    
    with torch.no_grad():
        xy_tensor = torch.tensor(xy, dtype=torch.float32).to(device)
        predict = model(xy_tensor).cpu().numpy()

    u_predict = predict[:, 0].reshape((resolution, resolution))
    v_predict = predict[:, 1].reshape((resolution, resolution))
    p_predict = predict[:, 2].reshape((resolution, resolution))

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("u")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ac1 = ax1.contourf(x_grid, y_grid, u_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("v")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ac2 = ax2.contourf(x_grid, y_grid, v_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("p")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ac3 = ax3.contourf(x_grid, y_grid, p_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac3, ax=ax3)

    plt.tight_layout()
    
    os.makedirs("images", exist_ok=True)
    plt.savefig(
        "images/kovasznay_epochs_{}_lr_{}.png".format(
            config["epochs"], config["optimizer"]["initial_lr"]
        )
    )
    plt.close()