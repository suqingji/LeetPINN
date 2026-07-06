"""visualization and calculation functions for PyTorch"""
import time
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torch

def print_log(*args):
    print(*args)

def visual(model, epochs=1, resolution=100, device="cpu"):
    """visulization of ex/ey/hz"""
    model.eval()
    t_flat = np.linspace(0, 1, resolution)
    x_flat = np.linspace(-1, 1, resolution)
    t_grid, x_grid = np.meshgrid(t_flat, x_flat)
    x = x_grid.reshape((-1, 1))
    t = t_grid.reshape((-1, 1))
    
    xt = torch.tensor(np.concatenate((x, t), axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        u_predict = model(xt).cpu().numpy()
        
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.scatter(t, x, c=u_predict, cmap=plt.cm.rainbow)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        xt_cs = torch.tensor(np.stack([x_flat, np.full(x_flat.shape, t_cs)], axis=-1), dtype=torch.float32).to(device)
        with torch.no_grad():
            u = model(xt_cs).cpu().numpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        
    plt.tight_layout()
    import os
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/{epochs}-result.jpg')
    plt.close()


def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    error = label - prediction
    l2_error = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size, device):
    '''calculate the prediction respect to the given inputs'''
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()
    model.eval()
    
    index = 0
    with torch.no_grad():
        while index < inputs.shape[0]:
            index_end = min(index + batch_size, inputs.shape[0])
            test_batch = torch.tensor(inputs[index: index_end, :], dtype=torch.float32).to(device)
            prediction[index: index_end, :] = model(test_batch).cpu().numpy()
            index = index_end

    print_log("    predict total time: {:.3f} ms".format((time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size, device):
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, batch_size, device)
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    print_log("    l2_error: ", l2_error)
    print_log("==================================================================================================")