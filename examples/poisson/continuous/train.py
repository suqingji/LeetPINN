"""Training."""
import os
import time
import argparse
import yaml
import torch

from src.model import create_model
from src.dataset import create_dataset
from src.poisson import PoissonPINN
from src.utils import calculate_l2_error, draw2d

def parse_args():
    parser = argparse.ArgumentParser(description="poisson_pytorch")
    parser.add_argument("--geom_name", type=str, default="disk")
    parser.add_argument("--ckpt_dir", default="./ckpt")
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--config_file_path", type=str, default="./configs/poisson_cfg.yaml")
    return parser.parse_args()

def train(geom_name, file_cfg, ckpt_dir, n_epochs):
    with open(file_cfg, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 建立数据集
    train_loader, n_dim = create_dataset(geom_name, config)

    # 2. 建立模型和PINN问题求解器
    model = create_model(**config["model"][f"{n_dim}d"]).to(device)
    problem = PoissonPINN(model, n_dim).to(device)

    # 3. 配置优化器和学习率
    params = list(model.parameters()) + list(problem.loss_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=config["optimizer"]["lr_max"])
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["optimizer"]["lr_max"],
        epochs=n_epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=config["optimizer"].get("pct_start", 0.3)
    )

    keep_ckpt_max = config.get("keep_checkpoint_max", 2)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4. 开始训练
    for i_epoch in range(1, 1 + n_epochs):
        model.train()
        for i_step, (pde_data, bc_data) in enumerate(train_loader):
            local_time_beg = time.time()
            pde_data, bc_data = pde_data.to(device), bc_data.to(device)

            optimizer.zero_grad()
            total_loss, loss_pde, loss_bc = problem(pde_data, bc_data)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if i_step % 50 == 0 or i_step + 1 == steps_per_epoch:
                epoch_seconds = (time.time() - local_time_beg) * 1000
                print(f"epoch: {i_epoch}, loss: {total_loss.item():>7f}, "
                      f"time elapsed: {epoch_seconds:.1f}ms [{i_step + 1}/{steps_per_epoch}]")

        # 保存权重
        save_name = os.path.join(ckpt_dir, f"{geom_name}_{n_dim}d_{i_epoch % keep_ckpt_max}.pth")
        torch.save(model.state_dict(), save_name)

def test(geom_name, checkpoint, file_cfg, n_samps):
    with open(file_cfg, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, n_dim_ = create_dataset(geom_name, config, n_samps)

    model_ = create_model(**config["model"][f"{n_dim_}d"]).to(device)
    model_.load_state_dict(torch.load(checkpoint, map_location=device))

    calculate_l2_error(model_, test_loader, n_dim_, device)
    
    if n_dim_ == 2: # 二维时进行画图展示
        draw2d(model_, 100, device)

if __name__ == "__main__":
    torch.manual_seed(123456)
    print(f"pid: {os.getpid()}")
    args = parse_args()
    
    time_beg = time.time()
    train(args.geom_name, args.config_file_path, args.ckpt_dir, args.n_epochs)
    print(f"End-to-End total time: {time.time() - time_beg} s")
    
    # 提取最后保存的模型权重进行测试和画图
    ckpt = os.path.join(args.ckpt_dir, f"{args.geom_name}_2d_0.pth")
    if os.path.exists(ckpt):
        test(args.geom_name, ckpt, args.config_file_path, 5000)