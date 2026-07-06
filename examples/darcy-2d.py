import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 保证实验可重复性
torch.manual_seed(123456)
np.random.seed(123456)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {device}")

# ==========================================
# 1. 定义带有残差连接的全连接神经网络
# 对应 MindSpore 中的 FCSequential(residual=True)
# ==========================================
class ResBlock(nn.Module):
    """残差块：增强深层网络中物理梯度的稳定传播"""
    def __init__(self, hidden_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return x + self.fc(x)

class DarcyNetwork(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, neurons=128):
        super(DarcyNetwork, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(in_channels, neurons),
            nn.Tanh()
        )
        
        # 使用多层残差块搭建深层网络
        self.hidden_layers = nn.ModuleList([ResBlock(neurons) for _ in range(4)])
        self.output_layer = nn.Linear(neurons, out_channels)

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

# ==========================================
# 2. 2D Darcy 问题求解器封装 (对应原文件的 Darcy2D 类)
# ==========================================
class Darcy2DPINN():
    def __init__(self, model, x_pde, y_pde, x_bc, y_bc, lr=1e-3):
        self.model = model.to(device)
        
        # 将数据集保存为纯数据张量，避免 __init__ 绑定梯度导致显存泄漏
        self.x_pde = torch.tensor(x_pde).float().to(device)
        self.y_pde = torch.tensor(y_pde).float().to(device)
        self.x_bc = torch.tensor(x_bc).float().to(device)
        self.y_bc = torch.tensor(y_bc).float().to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def force_function(self, x, y):
        """对应控制方程中的 forcing function f"""
        return 8 * torch.pi**2 * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)

    def compute_pde_loss(self):
        """计算域内偏微分方程残差"""
        # 动态创建局部叶子节点，彻底阻断跨 epoch 的计算图泄漏 (非常关键！)
        x = self.x_pde.clone().requires_grad_(True)
        y = self.y_pde.clone().requires_grad_(True)
        
        inputs = torch.cat([x, y], dim=1)
        outputs = self.model(inputs)
        
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # 使用自动微分计算关于 x, y 的偏导数
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        
        f = self.force_function(x, y)
        
        # Darcy 方程组物理残差
        loss_1 = u_x + v_y - f
        loss_2 = u + p_x
        loss_3 = v + p_y
        
        pde_loss = self.loss_fn(loss_1, torch.zeros_like(loss_1)) + \
                   self.loss_fn(loss_2, torch.zeros_like(loss_2)) + \
                   self.loss_fn(loss_3, torch.zeros_like(loss_3))
                   
        return pde_loss

    def compute_bc_loss(self):
        """计算边界条件残差"""
        inputs = torch.cat([self.x_bc, self.y_bc], dim=1)
        outputs = self.model(inputs)
        
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # 组装精确的边界条件约束值
        u_exact = -2 * torch.pi * torch.cos(2 * torch.pi * self.x_bc) * torch.cos(2 * torch.pi * self.y_bc)
        v_exact =  2 * torch.pi * torch.sin(2 * torch.pi * self.x_bc) * torch.sin(2 * torch.pi * self.y_bc)
        p_exact =  torch.sin(2 * torch.pi * self.x_bc) * torch.cos(2 * torch.pi * self.y_bc)
        
        bc_loss = self.loss_fn(u, u_exact) + \
                  self.loss_fn(v, v_exact) + \
                  self.loss_fn(p, p_exact)
                  
        return bc_loss

    def train(self, epochs):
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            
            pde_loss = self.compute_pde_loss()
            bc_loss = self.compute_bc_loss()
            total_loss = pde_loss + bc_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 200 == 0:
                print(f"Epoch: {epoch:04d} | Total Loss: {total_loss.item():.4e} | PDE Loss: {pde_loss.item():.4e} | BC Loss: {bc_loss.item():.4e}")
                
        print(f"训练结束，总耗时: {time.time() - start_time:.2f} 秒")

    def predict(self, x, y):
        self.model.eval()
        x_t = torch.tensor(x).float().to(device)
        y_t = torch.tensor(y).float().to(device)
        inputs = torch.cat([x_t, y_t], dim=1)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.cpu().numpy()

# ==========================================
# 3. 数据集构建与执行入口
# ==========================================
def generate_data(n_pde=8192, n_bc=8192):
    """根据域信息均匀随机生成域内配点与边界点"""
    # 域内配点 (0, 1) x (0, 1)
    x_pde = np.random.rand(n_pde, 1)
    y_pde = np.random.rand(n_pde, 1)
    
    x_bc = []
    y_bc = []
    
    # 构造 y = 0 和 y = 1 上下边界点
    x_tb = np.random.rand(n_bc // 2, 1)
    y_tb = np.random.choice([0.0, 1.0], size=(n_bc // 2, 1))
    x_bc.append(x_tb)
    y_bc.append(y_tb)
    
    # 构造 x = 0 和 x = 1 左右边界点
    y_lr = np.random.rand(n_bc // 2, 1)
    x_lr = np.random.choice([0.0, 1.0], size=(n_bc // 2, 1))
    x_bc.append(x_lr)
    y_bc.append(y_lr)
    
    x_bc = np.vstack(x_bc)
    y_bc = np.vstack(y_bc)
    
    return x_pde, y_pde, x_bc, y_bc

def main():
    # 1. 准备数据采样
    x_pde, y_pde, x_bc, y_bc = generate_data(n_pde=8000, n_bc=2000)
    
    # 2. 实例化网络与求解器
    model = DarcyNetwork(in_channels=2, out_channels=3, neurons=128)
    solver = Darcy2DPINN(model, x_pde, y_pde, x_bc, y_bc, lr=1e-3)
    
    # 3. 训练模型 (对标 MindSpore 示例跑约 4000 个 Epoch)
    solver.train(epochs=4000)
    
    # 4. 推理与压力场 p 可视化
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    x_test = X.flatten()[:, None]
    y_test = Y.flatten()[:, None]
    
    preds = solver.predict(x_test, y_test)
    # 取网络输出的第三个通道为压力场 p
    P_pred = preds[:, 2].reshape(100, 100) 
    
    # 提供精确解析压力场作为比对: p = sin(2*pi*x)*cos(2*pi*y)
    P_exact = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    
    # 画图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, P_pred, cmap='jet', shading='auto')
    plt.colorbar()
    plt.title('Predicted Pressure (PINN)')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, P_exact, cmap='jet', shading='auto')
    plt.colorbar()
    plt.title('Exact Pressure')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()