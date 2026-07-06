import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

# 保证实验可重复性
torch.manual_seed(123456)
np.random.seed(123456)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {device}")



def calculate_l2_error(pred, exact):
    """
    计算相对 L2 误差: ||pred - exact||_2 / ||exact||_2
    """
    error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
    return error

def visual_and_l2_error(model, device, t_eval=1.0, x_loc=0.5):
    """
    流场云图可视化与取线 L2 误差对比计算
    
    参数:
    model: 训练好的 PyTorch PINN 模型
    device: 计算设备 (cpu/cuda)
    t_eval: 评估流场的特定时间点 t
    x_loc: 截线提取的位置 (默认 x=0.5)
    """
    model.eval()
    
    # ==========================================
    # 1. 绘制流场云图 (Flow Field Contour)
    # ==========================================
    # 构造全域评估网格
    x_range = np.linspace(-1.0, 2.0, 200)
    y_range = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # 展平并添加时间维度 t
    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    t_flat = np.full_like(x_flat, t_eval)
    
    # 转换为 Tensor 进行推理
    x_t = torch.tensor(x_flat).float().to(device)
    y_t = torch.tensor(y_flat).float().to(device)
    t_t = torch.tensor(t_flat).float().to(device)
    
    inputs = torch.cat([x_t, y_t, t_t], dim=1)
    with torch.no_grad():
        outputs = model(inputs).cpu().numpy()
        
    u_pred = outputs[:, 0].reshape(X.shape)
    v_pred = outputs[:, 1].reshape(X.shape)
    p_pred = outputs[:, 2].reshape(X.shape)
    
    # 画流场云图
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(f"Predicted Flow Field at t = {t_eval}", fontsize=14)
    
    # U 速度
    ax1 = fig.add_subplot(1, 3, 1)
    c1 = ax1.pcolormesh(X, Y, u_pred, cmap='jet', shading='auto')
    ax1.set_title('Velocity U')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(c1, ax=ax1)
    
    # V 速度
    ax2 = fig.add_subplot(1, 3, 2)
    c2 = ax2.pcolormesh(X, Y, v_pred, cmap='jet', shading='auto')
    ax2.set_title('Velocity V')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(c2, ax=ax2)
    
    # 压力 P
    ax3 = fig.add_subplot(1, 3, 3)
    c3 = ax3.pcolormesh(X, Y, p_pred, cmap='jet', shading='auto')
    ax3.set_title('Pressure P')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(c3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. 提取截线对比与 L2 误差 (Line Extraction)
    # ==========================================
    # 在 x = x_loc 处取一条沿 Y 轴的线段
    y_line = np.linspace(-1.0, 1.0, 200)[:, None]
    x_line = np.full_like(y_line, x_loc)
    t_line = np.full_like(y_line, t_eval)
    
    x_line_t = torch.tensor(x_line).float().to(device)
    y_line_t = torch.tensor(y_line).float().to(device)
    t_line_t = torch.tensor(t_line).float().to(device)
    
    line_inputs = torch.cat([x_line_t, y_line_t, t_line_t], dim=1)
    with torch.no_grad():
        line_preds = model(line_inputs).cpu().numpy()
        
    u_line_pred = line_preds[:, 0:1]
    v_line_pred = line_preds[:, 1:2]
    
    # ----------------------------------------------------
    # 【注】此处使用占位函数模拟真实 CFD 数据。
    # 实际工程中，请将此处替换为从真实验证集读取的精确解 (Exact Solution)。
    # ----------------------------------------------------
    u_exact = 1.0 - (y_line**2) # 模拟抛物线型速度分布真值
    v_exact = 0.1 * np.sin(np.pi * y_line) # 模拟某种横向扰动真值
    
    # 计算相对 L2 误差
    l2_u = calculate_l2_error(u_line_pred, u_exact)
    l2_v = calculate_l2_error(v_line_pred, v_exact)
    print("="*50)
    print(f"截线 x = {x_loc}, t = {t_eval} 处的相对 L2 误差:")
    print(f"U Velocity L2 Error: {l2_u:.4e}")
    print(f"V Velocity L2 Error: {l2_v:.4e}")
    print("="*50)
    
    # 绘制取线对比折线图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_line, u_exact, 'b-', label='Exact U', linewidth=2)
    plt.plot(y_line, u_line_pred, 'r--', label='PINN U', linewidth=2)
    plt.xlabel('y')
    plt.ylabel('u')
    plt.title(f'U Profile at x={x_loc}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_line, v_exact, 'b-', label='Exact V', linewidth=2)
    plt.plot(y_line, v_line_pred, 'r--', label='PINN V', linewidth=2)
    plt.xlabel('y')
    plt.ylabel('v')
    plt.title(f'V Profile at x={x_loc}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 1. 定义多层全连接网络
# ==========================================
class NavierStokesNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, layers=6, neurons=128):
        super(NavierStokesNetwork, self).__init__()
        
        modules = []
        modules.append(nn.Linear(in_channels, neurons))
        modules.append(nn.Tanh())
        
        for _ in range(layers - 2):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.Tanh())
            
        modules.append(nn.Linear(neurons, out_channels))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 不确定性权重多任务学习 (MTLWeightedLoss)
# 对标原文件中的 MTLWeightedLossCell
# ==========================================
class MTLWeightedLoss(nn.Module):
    def __init__(self, num_losses=3):
        super(MTLWeightedLoss, self).__init__()
        # 初始化可学习的对数方差参数 (log_sigma^2)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        """
        基于不确定性的损失加权: L = sum( L_i * exp(-log_var_i) + log_var_i )
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

# ==========================================
# 3. NavierStokes2D 求解器封装
# ==========================================
class NavierStokes2DPINN():
    def __init__(self, model, re=100.0, lr=1e-3):
        self.model = model.to(device)
        self.re = re
        
        # 多任务自适应损失模块 (3个Loss: PDE, IC, BC)
        self.mtl_loss = MTLWeightedLoss(num_losses=3).to(device)
        
        # 优化器：联合优化网络参数和 Loss 权重参数
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.mtl_loss.parameters()), 
            lr=lr
        )
        self.mse = nn.MSELoss()

    def load_data(self, pde_data, ic_data, ic_label, bc_data, bc_label):
        """加载为普通的 GPU Tensor (剥离持久梯度防止显存泄漏)"""
        self.x_pde = torch.tensor(pde_data[:, 0:1]).float().to(device)
        self.y_pde = torch.tensor(pde_data[:, 1:2]).float().to(device)
        self.t_pde = torch.tensor(pde_data[:, 2:3]).float().to(device)
        
        self.x_ic = torch.tensor(ic_data[:, 0:1]).float().to(device)
        self.y_ic = torch.tensor(ic_data[:, 1:2]).float().to(device)
        self.t_ic = torch.tensor(ic_data[:, 2:3]).float().to(device)
        self.ic_label = torch.tensor(ic_label).float().to(device) # [u, v, p]
        
        self.x_bc = torch.tensor(bc_data[:, 0:1]).float().to(device)
        self.y_bc = torch.tensor(bc_data[:, 1:2]).float().to(device)
        self.t_bc = torch.tensor(bc_data[:, 2:3]).float().to(device)
        self.bc_label = torch.tensor(bc_label).float().to(device) # [u, v] (一般边界给定速度)

    def compute_pde_loss(self):
        # 动态创建局部叶子节点
        x = self.x_pde.clone().requires_grad_(True)
        y = self.y_pde.clone().requires_grad_(True)
        t = self.t_pde.clone().requires_grad_(True)
        
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.model(inputs)
        u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
        
        # 一阶导数
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        
        # 二阶导数
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        
        # 组装 N-S 方程残差
        f_u = u_t + u * u_x + v * u_y + p_x - (1.0 / self.re) * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - (1.0 / self.re) * (v_xx + v_yy)
        f_c = u_x + v_y # 连续性方程
        
        loss_pde = self.mse(f_u, torch.zeros_like(f_u)) + \
                   self.mse(f_v, torch.zeros_like(f_v)) + \
                   self.mse(f_c, torch.zeros_like(f_c))
        return loss_pde

    def compute_data_loss(self, x, y, t, label, check_p=True):
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.model(inputs)
        
        # 对于初始条件，通常有 u,v,p 的真值
        if check_p:
            return self.mse(outputs, label)
        # 对于边界条件，圆柱表面无滑移通常只约束 u, v = 0
        else:
            return self.mse(outputs[:, 0:2], label[:, 0:2])

    def train(self, epochs):
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            
            loss_pde = self.compute_pde_loss()
            loss_ic = self.compute_data_loss(self.x_ic, self.y_ic, self.t_ic, self.ic_label, check_p=True)
            loss_bc = self.compute_data_loss(self.x_bc, self.y_bc, self.t_bc, self.bc_label, check_p=False)
            
            # 使用自适应多任务学习合并 Loss
            total_loss = self.mtl_loss([loss_pde, loss_ic, loss_bc])
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch: {epoch:05d} | Total: {total_loss.item():.4e} "
                      f"| PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | BC: {loss_bc.item():.4e}")
                
        print(f"训练结束，总耗时: {time.time() - start_time:.2f} 秒")

# ==========================================
# 4. 数据生成器 (模拟外部数据集输入)
# ==========================================
def generate_mock_data(N_pde=5000, N_bc=1000, N_ic=1000):
    """
    占位数据生成器：在实际工程中，请替换为您提取的真实的圆柱网格点(x, y)及对应时序t数据。
    返回的维度规范: 数据[N, 3]代表(x,y,t) 标签[N, 3]代表(u,v,p)
    """
    # 模拟 PDE 配点
    pde_data = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[2.0, 1.0, 1.0], size=(N_pde, 3))
    
    # 模拟初始条件点 (t=0)
    ic_data = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[2.0, 1.0, 0.0], size=(N_ic, 3))
    ic_label = np.zeros((N_ic, 3)) # 此处需接入真值
    
    # 模拟边界条件点 (如圆柱表面及外边界)
    bc_data = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[2.0, 1.0, 1.0], size=(N_bc, 3))
    bc_label = np.zeros((N_bc, 2)) # 速度无滑移边界 u=0, v=0
    
    return pde_data, ic_data, ic_label, bc_data, bc_label

# ==========================================
# 5. 主执行程序
# ==========================================
def main():
    # 1. 准备数据
    pde_data, ic_data, ic_label, bc_data, bc_label = generate_mock_data()
    
    # 2. 建立模型
    model = NavierStokesNetwork(in_channels=3, out_channels=3, layers=6, neurons=128)
    
    # 3. 初始化求解器，设定雷诺数 Re=100
    solver = NavierStokes2DPINN(model, re=100.0, lr=1e-3)
    solver.load_data(pde_data, ic_data, ic_label, bc_data, bc_label)
    
    # 4. 模型训练 (跑小量 Epoch 作验证，实际通常需几万个 Epoch)
    solver.train(epochs=2000)

    # 5. 可视化与误差计算 (对标 MindSpore 的 visual 功能)
    # 取 t=1.0 时刻，并在圆柱尾流区域 x=0.5 处切片比对
    visual_and_l2_error(model, device, t_eval=1.0, x_loc=0.5)

if __name__ == '__main__':
    main()