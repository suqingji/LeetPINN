import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import torch

def plot_2d(u_label, u_predict, file_name=None):
    u_error = np.abs(u_label - u_predict)
    vmin = [u_label.min(), u_label.min(), u_error.min()]
    vmax = [u_label.max(), u_label.max(), u_error.max()]
    sub_titles = ["Reference", "Predict", "Error"]

    plt.rcParams['figure.figsize'] = [9.6, 3.2]
    fig = plt.figure()
    gs_ = gridspec.GridSpec(2, 6)
    slice_ = [gs_[0:2, 0:2], gs_[0:2, 2:4], gs_[0:2, 4:6]]
    
    for i, data in enumerate([u_label, u_predict, u_error]):
        ax_ = fig.add_subplot(slice_[i])
        img = ax_.imshow(data.T, vmin=vmin[i], vmax=vmax[i], cmap=plt.get_cmap("jet"), origin='lower')
        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)
    if file_name is None:
        plt.show()
    else:
        os.makedirs("./images", exist_ok=True)
        fig.savefig(os.path.join("./images", file_name))
    plt.close()

def visual(model, ds_test, device, n_samps_per_axis=100, file_name=None):
    mesh, label = ds_test[0].to(device), ds_test[1]
    model.eval()
    with torch.no_grad():
        pred = model(mesh).cpu().numpy()
    label = label.numpy()
    plot_2d(label.reshape(n_samps_per_axis, n_samps_per_axis),
            pred.reshape(n_samps_per_axis, n_samps_per_axis),
            file_name=file_name)

def calculate_l2_error(model, ds_test, device):
    mesh, label = ds_test[0].to(device), ds_test[1]
    model.eval()
    with torch.no_grad():
        pred = model(mesh).cpu().numpy().flatten()
    label = label.numpy().flatten()

    error_norm = np.linalg.norm(pred - label, ord=2)
    label_norm = np.linalg.norm(label, ord=2)
    relative_l2_error = error_norm / label_norm
    print(f"Relative L2 error: {relative_l2_error:>8.4f}")