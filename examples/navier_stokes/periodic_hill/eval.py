import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch

from src.model import FCSequential
from src.dataset import create_test_dataset

# 为了与 config 文件读取兼容，手动实现简单的 yaml 加载
def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

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

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.pcolor(x.T, y.T, prediction[:, :, 0].T, cmap='jet')
    plt.title("U prediction")
    
    plt.subplot(2, 2, 2)
    plt.pcolor(x.T, y.T, prediction[:, :, 1].T, cmap='jet')
    plt.title("V prediction")
    
    plt.subplot(2, 2, 3)
    plt.pcolormesh(x.T, y.T, label[:, :, 0].T, cmap='jet')
    plt.title("U ground truth")
    
    plt.subplot(2, 2, 4)
    plt.pcolormesh(x.T, y.T, label[:, :, 1].T, cmap='jet')
    plt.title("V ground truth")
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(epochs) + "_eval.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rans-pinns prediction (PyTorch)")
    parser.add_argument("--config_file_path", type=str, default="./configs/rans.yaml")
    args = parser.parse_args()
    print(f"pid:{os.getpid()}")

    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    summary_params = config["summary"]
    optim_params = config["optimizer"]

    # 自动探测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 实例化网络
    rans_model = FCSequential(in_channels=model_params["in_channels"],
                              out_channels=model_params["out_channels"],
                              layers=model_params["layers"],
                              neurons=model_params["neurons"],
                              residual=model_params["residual"],
                              act=model_params["activation"]).to(device)
                              
    # 加载测试数据
    inputs, labels = create_test_dataset(data_params["data_path"])

    # 加载已保存的模型权重
    load_ckpt_path = summary_params.get("load_ckpt_path", "./summary/ckpt/rans_1000.ckpt")
    if os.path.exists(load_ckpt_path):
        # map_location 确保了即使是在只有 CPU 的机器上也能加载 GPU 训练出来的权重
        rans_model.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        print(f"Successfully loaded checkpoint from {load_ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {load_ckpt_path}. Using random weights.")

    # 执行预测并出图
    epochs = optim_params.get("train_epochs", 1600)
    visual_path = summary_params.get("visual_dir", "./prediction_result")
    
    predict(rans_model, epochs, inputs, labels, device, visual_path)