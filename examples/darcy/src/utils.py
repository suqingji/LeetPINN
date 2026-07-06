"""utility functions (PyTorch)"""
import os
import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import torch

from .dataset import create_test_dataset

plt.rcParams["figure.dpi"] = 300

def visual(model, config, device):
    """visual result of model prediction and ground-truth"""
    name = "result"
    test_input, label = create_test_dataset(config)
    visual_input = test_input.reshape(-1, config["model"]["in_channels"])
    visual_label = label.reshape(-1, config["model"]["out_channels"])
    prediction = np.zeros(label.shape)
    
    index = 0
    model.eval()
    with torch.no_grad():
        while index < len(visual_input):
            index_end = min(index + config["data"]["train"]["batch_size"], len(visual_input))
            test_batch = torch.tensor(visual_input[index:index_end, :], dtype=torch.float32).to(device)
            predict = model(test_batch).cpu().numpy()
            for i in range(config["model"]["out_channels"]):
                prediction[index:index_end, i] = predict[:, i]
            index = index_end

    visual_fn(visual_label, prediction.reshape(visual_label.shape), config["summary"]["vision_dir"], name)

def visual_fn(label, predict, path, name):
    """visulization of ux/uy/p (Kept original logic)"""
    ux_min, ux_max = np.percentile(label[:, 0], [0.5, 99.5])
    uy_min, uy_max = np.percentile(label[:, 1], [0.5, 99.5])
    p_min, p_max = np.percentile(label[:, 2], [0.5, 99.5])

    min_list = [ux_min, uy_min, p_min]
    max_list = [ux_max, uy_max, p_max]

    output_names = ["ux", "uy", "p"]
    if not os.path.isdir(path):
        os.makedirs(path)

    ux_error_2d = np.abs(predict[:, 0] - label[:, 0])
    uy_error_2d = np.abs(predict[:, 1] - label[:, 1])
    p_error_2d = np.abs(predict[:, 2] - label[:, 2])

    label_2d = [label[:, 0], label[:, 1], label[:, 2]]
    pred_2d = [predict[:, 0], predict[:, 1], predict[:, 2]]
    error_2d = [ux_error_2d, uy_error_2d, p_error_2d]

    lpe_2d = [label_2d, pred_2d, error_2d]
    lpe_names = ["label", "predict", "error"]

    fig = plt.figure()
    grid_spec = gridspec.GridSpec(3, 3)
    grid_spec_idx = int(0)

    for i, data_2d in enumerate(lpe_2d):
        for j, data in enumerate(data_2d):
            subfig = fig.add_subplot(grid_spec[grid_spec_idx])
            grid_spec_idx += 1

            if lpe_names[i] == "error":
                img = subfig.imshow(data.T.reshape(101, 101), vmin=0, vmax=1, cmap=plt.get_cmap("jet"), origin="lower")
            else:
                img = subfig.imshow(data.T.reshape(101, 101), vmin=min_list[j], vmax=max_list[j], cmap=plt.get_cmap("jet"), origin="lower")

            subfig.set_title(output_names[j] + " " + lpe_names[i], fontsize=4)
            plt.xticks(size=4)
            plt.yticks(size=4)

            aspect = 20
            pad_fraction = 0.5
            divider = make_axes_locatable(subfig)
            width = axes_size.AxesY(subfig, aspect=1 / aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            colorbar = plt.colorbar(img, cax=cax)
            colorbar.ax.tick_params(labelsize=4)

    grid_spec.tight_layout(fig, pad=0.4, w_pad=0.4, h_pad=0.4)
    fig.savefig(os.path.join(path, f"{name}.png"), format="png")
    plt.close()

def _calculate_error(label, prediction):
    error = label - prediction
    return np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

def calculate_l2_error(model, inputs, label, batch_size, device):
    label_shape = label.shape
    prediction = np.zeros(label_shape).reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()
    model.eval()
    index = 0
    with torch.no_grad():
        while index < inputs.shape[0]:
            index_end = min(index + batch_size, inputs.shape[0])
            test_batch = torch.tensor(inputs[index:index_end, :], dtype=torch.float32).to(device)
            prediction[index:index_end, :] = model(test_batch).cpu().numpy()
            index = index_end

    print(f"    predict total time: {(time.time() - time_beg) * 1000:.2f} ms")
    prediction = prediction.reshape(label_shape).reshape((-1, label_shape[1]))
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    print("    l2_error: ", l2_error)
    print("==================================================================================================")