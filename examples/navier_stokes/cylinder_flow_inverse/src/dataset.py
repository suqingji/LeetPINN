import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_test_dataset(config):
    """load labeled data for evaluation"""
    test_data_path = config["data"]["root_dir"]
    print("get dataset path: {}".format(test_data_path))
    paths = [test_data_path + '/eval_points.npy', test_data_path + '/eval_label.npy']
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_training_dataset(config):
    """create training dataset by loading local npy files"""
    data_path = config["data"]["root_dir"]
    print("train dataset path:", data_path)
    
    # 直接加载 npy 文件
    points = np.load(data_path + "/train_points.npy").astype(np.float32)
    labels = np.load(data_path + "/train_label.npy").astype(np.float32)
    
    dataset = TensorDataset(torch.tensor(points), torch.tensor(labels))
    
    # 保持和 MindSpore 相同的批处理行为
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["train"]["batch_size"],
        shuffle=True,
        drop_last=True
    )
    
    return dataloader