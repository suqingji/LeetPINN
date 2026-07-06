"""Training."""
import os
import time
import argparse
import yaml
import torch
import numpy as np

from src.dataset import create_train_dataset, create_test_dataset
from src.model import MultiScaleFCSequential
from src.poisson import PoissonPINN
from src.utils import calculate_l2_error, visual

def set_seed(seed=123456):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(file_cfg, ckpt_dir, n_epochs):
    with open(file_cfg, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集
    train_loader = create_train_dataset(config)
    ds_test = create_test_dataset(config)

    # 建立多尺度模型
    model = MultiScaleFCSequential(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        layers=config['model']['layers'],
        neurons=config['model']['neurons'],
        residual=True,
        act=config['model']['activation'],
        num_scales=config['model']['num_scales'],
        scale_factor=2.0,
        input_scale=[10., 10.]
    ).to(device)
    print(model)

    # PINN 损失计算工具
    problem = PoissonPINN(model).to(device)

    # 配置优化器与多步衰减学习率
    params = list(model.parameters()) + list(problem.loss_fn.parameters())
    lr_init = config["optimizer"]["initial_lr"]
    optimizer = torch.optim.Adam(params, lr=lr_init)

    # 等效复现 MindSpore 版 piecewise_constant_lr 策略
    steps_per_epoch = len(train_loader)
    milestones = [int(n_epochs * 0.4), int(n_epochs * 0.6), int(n_epochs * 0.8)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    keep_ckpt_max = config.get('keep_checkpoint_max', 2)
    os.makedirs(ckpt_dir, exist_ok=True)

    for i_epoch in range(1, 1 + n_epochs):
        local_time_beg = time.time()
        model.train()
        
        for i_step, (pde_data, bc_data, src_data) in enumerate(train_loader):
            pde_data, bc_data, src_data = pde_data.to(device), bc_data.to(device), src_data.to(device)
            
            optimizer.zero_grad()
            loss = problem(pde_data, bc_data, src_data)
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"epoch: {i_epoch} train loss: {loss.item():.8f} epoch time: {time.time() - local_time_beg:.2f}s")

        # 保存权重
        save_name = os.path.join(ckpt_dir, f"epoch-{i_epoch % keep_ckpt_max}.pth")
        torch.save(model.state_dict(), save_name)

        # 定期评估与画图
        if i_epoch % 5 == 1 or i_epoch == n_epochs:
            calculate_l2_error(model, ds_test, device)
            visual(model, ds_test, device, file_name=f"epoch-{i_epoch}_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="poisson point source")
    parser.add_argument('--ckpt_dir', default='./ckpt')
    parser.add_argument('--n_epochs', default=250, type=int)
    parser.add_argument("--config_file_path", type=str, default="./poisson_cfg.yaml")
    args = parser.parse_args()

    set_seed(123456)
    print(f'pid: {os.getpid()}')
    
    time_beg = time.time()
    train(args.config_file_path, args.ckpt_dir, args.n_epochs)
    print(f"End-to-End total time: {time.time() - time_beg:.1f} s")