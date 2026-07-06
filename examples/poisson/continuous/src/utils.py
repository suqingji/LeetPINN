import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class AnalyticSolution(nn.Module):
    def __init__(self, n_dim):
        super(AnalyticSolution, self).__init__()
        self.factor = 1.0 / (16.0 * n_dim * math.pi * math.pi)

    def forward(self, x):
        return self.factor * torch.prod(torch.sin(4 * math.pi * x), dim=1, keepdim=True)

def relative_l2(x, y):
    return torch.sqrt(torch.mean(torch.square(x - y))) / torch.sqrt(torch.mean(torch.square(y)))

def calculate_l2_error(model, ds_test_loader, n_dim, device):
    model.eval()
    solution = AnalyticSolution(n_dim).to(device)
    
    with torch.no_grad():
        for x_domain, x_bc in ds_test_loader:
            x_domain, x_bc = x_domain.to(device), x_bc.to(device)
            
            y_pred_domain = model(x_domain)
            y_test_domain = solution(x_domain)

            y_pred_bc = model(x_bc)
            y_test_bc = solution(x_bc)
            
            print("Relative L2 error (domain): {:.4f}".format(relative_l2(y_pred_domain, y_test_domain).item()))
            print("Relative L2 error (bc): {:.4f}".format(relative_l2(y_pred_bc, y_test_bc).item()))
            print("")
            break  # 全局采样一次即可

def draw2d(model, nmaps, device):
    '''draw the cloud map'''
    x = np.linspace(0., 1., nmaps)
    y = np.linspace(0., 1., nmaps)
    x, y = np.meshgrid(x, y)
    x_flat = np.reshape(x, (nmaps*nmaps, 1))
    y_flat = np.reshape(y, (nmaps*nmaps, 1))
    z = np.concatenate((x_flat, y_flat), axis=1)
    
    z_tensor = torch.tensor(z, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_z = model(z_tensor).cpu().numpy()

    plt.figure()
    plt.subplot(111)
    plt.scatter(x_flat, y_flat, c=pred_z, cmap=plt.cm.rainbow, vmin=min(pred_z), vmax=max(pred_z))
    plt.text(0.1, 0.9, r'DNN', {'color': 'b', 'fontsize': 20}, transform=plt.gca().transAxes)
    plt.colorbar()
    plt.title("dirichlet")
    plt.show()