import collections
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams['figure.dpi'] = 300

def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy on 5 variables'''
    errors = collections.namedtuple("PeriodicHillError", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p", "l2_error_k"])
    
    # 标签 (label) 的维度为 6 (u, v, p, uu, uv, vv)
    # 预测 (prediction) 的维度为 5 (u, v, p, k, omega)
    error_u = label[..., 0] - prediction[..., 0]
    error_v = label[..., 1] - prediction[..., 1]
    error_p = label[..., 2] - prediction[..., 2]

    # 利用 uu 和 vv 计算近似的真实 k
    k_true = 0.5 * (label[..., 3] + label[..., 5])
    error_k = k_true - prediction[..., 3]

    l2_error_u = np.sqrt(np.sum(np.square(error_u))) / (np.sqrt(np.sum(np.square(label[..., 0]))) + 1e-8)
    l2_error_v = np.sqrt(np.sum(np.square(error_v))) / (np.sqrt(np.sum(np.square(label[..., 1]))) + 1e-8)
    l2_error_p = np.sqrt(np.sum(np.square(error_p))) / (np.sqrt(np.sum(np.square(label[..., 2]))) + 1e-8)
    l2_error_k = np.sqrt(np.sum(np.square(error_k))) / (np.sqrt(np.sum(np.square(k_true))) + 1e-8)

    l2_error = np.sqrt(np.sum(np.square(error_u) + np.square(error_v) + np.square(error_p) + np.square(error_k))) / \
               np.sqrt(np.sum(np.square(label[..., 0]) + np.square(label[..., 1]) + np.square(label[..., 2]) + np.square(k_true)) + 1e-8)
               
    return errors(l2_error, l2_error_u, l2_error_v, l2_error_p, l2_error_k)

def _get_prediction(model, inputs, label_shape, config, device):
    '''calculate the prediction respect to the given inputs'''
    output_size = config['model']['out_channels'] # 现在为 5
    input_size = config['model']['in_channels']

    prediction = np.zeros((inputs.shape[0], output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()
    index = 0
    model.eval()
    with torch.no_grad():
        while index < inputs.shape[0]:
            index_end = min(index + config["data"]['batch_size'], inputs.shape[0])
            test_batch = torch.tensor(inputs[index: index_end, :], dtype=torch.float32).to(device)
            prediction[index: index_end, :] = model(test_batch).cpu().numpy()
            index = index_end

    print("    predict total time: {} ms".format((time.time() - time_beg)*1000))
    prediction = prediction.reshape((-1, output_size))
    return prediction

def calculate_l2_error(model, inputs, label, config, device):
    prediction = _get_prediction(model, inputs, label.shape, config, device)
    
    label_reshaped = label.reshape((-1, 6))
    l2_errors = _calculate_error(label_reshaped, prediction)
    
    print("    l2_error, U: {:.6f}, V: {:.6f}, P: {:.6f}, K: {:.6f}".format(
        l2_errors.l2_error_u, l2_errors.l2_error_v, l2_errors.l2_error_p, l2_errors.l2_error_k))
    print("    Total L2 Error: {:.6f}".format(l2_errors.l2_error))
    print("==================================================================================================")

def predict(model, epochs, input_data, label, device, path="./prediction_result"):
    """visulization of u/v/k"""
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        prediction = model(inputs_tensor).cpu().numpy()

    x = input_data[:, 0].reshape((300, 700))
    y = input_data[:, 1].reshape((300, 700))

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    k_true = 0.5 * (label[:, 3] + label[:, 5]).reshape((300, 700))
    label_u = label[:, 0].reshape((300, 700))
    label_v = label[:, 1].reshape((300, 700))

    pred_u = prediction[:, 0].reshape((300, 700))
    pred_v = prediction[:, 1].reshape((300, 700))
    pred_k = prediction[:, 3].reshape((300, 700))

    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.contourf(x.T, y.T, pred_u.T, levels=100, cmap='jet')
    plt.title("U prediction")
    plt.colorbar()
    
    plt.subplot(3, 2, 2)
    plt.contourf(x.T, y.T, label_u.T, levels=100, cmap='jet')
    plt.title("U ground truth")
    plt.colorbar()

    plt.subplot(3, 2, 3)
    plt.contourf(x.T, y.T, pred_v.T, levels=100, cmap='jet')
    plt.title("V prediction")
    plt.colorbar()
    
    plt.subplot(3, 2, 4)
    plt.contourf(x.T, y.T, label_v.T, levels=100, cmap='jet')
    plt.title("V ground truth")
    plt.colorbar()

    plt.subplot(3, 2, 5)
    plt.contourf(x.T, y.T, pred_k.T, levels=100, cmap='jet')
    plt.title("TKE (k) prediction")
    plt.colorbar()
    
    plt.subplot(3, 2, 6)
    plt.contourf(x.T, y.T, k_true.T, levels=100, cmap='jet')
    plt.title("TKE (k) approx truth")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(path, str(epochs) + ".png"))
    plt.close()