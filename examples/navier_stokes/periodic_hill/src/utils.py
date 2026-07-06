import collections
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams['figure.dpi'] = 300

def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    errors = collections.namedtuple("PeriodicHillError", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p",
                                                          "l2_error_uu", "l2_error_uv", "l2_error_vv"])
    error = label - prediction
    # x, y, u, v, p, uu, uv, vv, rho, nu
    l2_error_u = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error_uu = np.sqrt(np.sum(np.square(error[..., 3]))) / np.sqrt(np.sum(np.square(label[..., 3])))
    l2_error_uv = np.sqrt(np.sum(np.square(error[..., 4]))) / np.sqrt(np.sum(np.square(label[..., 4])))
    l2_error_vv = np.sqrt(np.sum(np.square(error[..., 5]))) / np.sqrt(np.sum(np.square(label[..., 5])))

    l2_error = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
    return errors(l2_error, l2_error_u, l2_error_v, l2_error_p, l2_error_uu, l2_error_uv, l2_error_vv)

def _get_prediction(model, inputs, label_shape, config, device):
    '''calculate the prediction respect to the given inputs'''
    output_size = config['model']['out_channels']
    input_size = config['model']['in_channels']

    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
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
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, output_size))
    return prediction

def calculate_l2_error(model, inputs, label, config, device):
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, config, device)
    output_size = config['model']['out_channels']
    label = label.reshape((-1, output_size))
    l2_errors = _calculate_error(label, prediction)
    
    print("    l2_error, U: ", l2_errors.l2_error_u, ", V: ", l2_errors.l2_error_v, ", P: ", l2_errors.l2_error_p)
    print("    l2_error, uu: ", l2_errors.l2_error_uu, ", uv: ", l2_errors.l2_error_uv, ", vv: ", l2_errors.l2_error_vv,
          ", Total: ", l2_errors.l2_error)
    print("==================================================================================================")

def predict(model, epochs, input_data, label, device, path="./prediction_result"):
    """visulization of u/v/p"""
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        prediction = model(inputs_tensor).cpu().numpy()

    x = input_data[:, 0].reshape((300, 700))
    y = input_data[:, 1].reshape((300, 700))

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    _, output_size = label.shape
    label = label.reshape((300, 700, output_size))
    prediction = prediction.reshape((300, 700, output_size))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.pcolor(x.T, y.T, prediction[:, :, 0].T)
    plt.title("U prediction")
    plt.subplot(2, 2, 2)
    plt.pcolor(x.T, y.T, prediction[:, :, 1].T)
    plt.title("V prediction")
    plt.subplot(2, 2, 3)
    plt.pcolormesh(x.T, y.T, label[:, :, 0].T)
    plt.title("U ground truth")
    plt.subplot(2, 2, 4)
    plt.pcolormesh(x.T, y.T, label[:, :, 1].T)
    plt.title("V ground truth")
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(epochs) + ".png"))
    plt.close()