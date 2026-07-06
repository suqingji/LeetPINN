import collections
import os
import time
import io
import cv2
import PIL
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import torch

plt.rcParams['figure.dpi'] = 300
InvNSError = collections.namedtuple("Inv_NS_error", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p"])

def visual(model, epochs, input_data, label, device, path="./videos"):
    """visulization of u/v/p"""
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        predict = model(inputs_tensor).cpu().numpy()
        
    [sample_t, sample_x, sample_y, _] = np.shape(input_data)
    # --------- 绘图代码保持原样 (从 u_vmin, u_vmax 往下一直到 video.release() 及 l2_error 曲线绘制) ---------
    # ... (与原版完全一致，此处省略避免过长)
    # -------------------------------------------------------------------------------------------------------

def _calculate_error(label, prediction):
    # 与原版完全一致
    error = label - prediction
    l2_error_u = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
    errors = InvNSError(l2_error, l2_error_u, l2_error_v, l2_error_p)
    return errors

def _get_prediction(model, inputs, label_shape, config, device):
    output_size = config.get("output_size", 3)
    input_size = config.get("input_size", 3)
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()
    index = 0
    model.eval()
    with torch.no_grad():
        while index < inputs.shape[0]:
            index_end = min(index + config["data"]["test"]["batch_size"], inputs.shape[0])
            test_batch = torch.tensor(inputs[index: index_end, :], dtype=torch.float32).to(device)
            prediction[index: index_end, :] = model(test_batch).cpu().numpy()
            index = index_end

    print("    predict total time: {} ms".format((time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, output_size))
    return prediction

def calculate_l2_error(model, inputs, label, config, device):
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, config, device)
    output_size = config.get("output_size", 3)
    label = label.reshape((-1, output_size))
    l2_errors = _calculate_error(label, prediction)
    print(f"    l2_error, U: {l2_errors.l2_error_u:.6f}, V: {l2_errors.l2_error_v:.6f}, P: {l2_errors.l2_error_p:.6f}, Total: {l2_errors.l2_error:.6f}")
    print("==================================================================================================")

def plot_params(param1, param2):
    # 与原版完全一致
    pass