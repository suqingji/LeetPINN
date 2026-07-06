import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 确保实验可重复性
torch.manual_seed(1234)
np.random.seed(1234)

# 设备配置 (优先使用 GPU/CUDA 进行加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {device}")

# ==========================================
# 1. 定义全连接神经网络 (FNN)
# ==========================================
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh() # PINN 常用 Tanh，因为其二阶导数连续且平滑
        
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(self.activation)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        
        self.mlp = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.mlp(x)

# ==========================================
# 2. 定义 PINN 求解器
# ==========================================
class PhysicsInformedNN():
    def __init__(self, X_pde, X_ic, u_ic, X_bc, u_bc, layers, nu):
        # 内部变量
        self.nu = nu
        
        # 将数据转换为张量并加载到设备
        # requires_grad=True 是计算物理残差偏导数的关键
        # self.x_pde = torch.tensor(X_pde[:, 0:1], requires_grad=True).float().to(device)
        # self.t_pde = torch.tensor(X_pde[:, 1:2], requires_grad=True).float().to(device)
        # self.x_pde = torch.tensor(X_pde[:, 0:1]).float().to(device).requires_grad_(True)
        # self.t_pde = torch.tensor(X_pde[:, 1:2]).float().to(device).requires_grad_(True)
        self.x_pde = torch.tensor(X_pde[:, 0:1]).float().to(device)
        self.t_pde = torch.tensor(X_pde[:, 1:2]).float().to(device)
        
        self.x_ic = torch.tensor(X_ic[:, 0:1]).float().to(device)
        self.t_ic = torch.tensor(X_ic[:, 1:2]).float().to(device)
        self.u_ic = torch.tensor(u_ic).float().to(device)
        
        self.x_bc = torch.tensor(X_bc[:, 0:1]).float().to(device)
        self.t_bc = torch.tensor(X_bc[:, 1:2]).float().to(device)
        self.u_bc = torch.tensor(u_bc).float().to(device)
        
        # 实例化网络
        self.dnn = DNN(layers).to(device)
        
        # 优化器：Adam 用于快速收敛，L-BFGS 用于高精度微调
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=1000, 
            max_eval=1000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1e-9
        )
        self.iter = 0

    def net_u(self, x, t):
        """网络输出 u"""
        X = torch.cat([x, t], dim=1)
        return self.dnn(X)

    # def net_f(self, x, t):
    #     """使用自动微分计算偏微分方程残差 f"""
    #     u = self.net_u(x, t)
        
    #     # 计算 u 对 t 和 x 的一阶偏导
    #     u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    #     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        
    #     # 计算 u_x 对 x 的一阶偏导，即 u 对 x 的二阶偏导
    #     u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        
    #     # 构造 Burgers 方程残差: f = u_t + u*u_x - nu*u_xx
    #     f = u_t + u * u_x - self.nu * u_xx
    #     return f
    def net_f(self, x_in, t_in):
        """使用自动微分计算偏微分方程残差 f"""
        # 1. 动态创建局部叶子节点，彻底阻断跨 epoch 的计算图泄漏
        x = x_in.clone().requires_grad_(True)
        t = t_in.clone().requires_grad_(True)
        
        u = self.net_u(x, t)
        
        # 2. 使用 .sum() 触发反向传播，代码更简洁且内存更安全
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
        # 3. 计算二阶导数
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        
        # 4. 构造 Burgers 方程残差: f = u_t + u*u_x - nu*u_xx
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_func(self):
        """计算总体损失"""
        # 1. 初始条件 Loss
        u_pred_ic = self.net_u(self.x_ic, self.t_ic)
        loss_ic = torch.mean((self.u_ic - u_pred_ic) ** 2)
        
        # 2. 边界条件 Loss
        u_pred_bc = self.net_u(self.x_bc, self.t_bc)
        loss_bc = torch.mean((self.u_bc - u_pred_bc) ** 2)
        
        # 3. 物理残差 Loss
        f_pred = self.net_f(self.x_pde, self.t_pde)
        loss_pde = torch.mean(f_pred ** 2)
        
        # 总体 Loss (此处未采用自适应权重，简单等权相加)
        loss = loss_ic + loss_bc + loss_pde
        return loss

    def train_adam(self, epochs):
        print("开始 Adam 训练...")
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4e}')

    def train_lbfgs(self):
        print("开始 L-BFGS 微调 (L-BFGS 通常用于 PDE 求解的高精度逼近)...")
        self.dnn.train()
        
        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.iter += 1
            if self.iter % 100 == 0:
                print(f'L-BFGS Iter {self.iter}, Loss: {loss.item():.4e}')
            return loss
            
        self.optimizer_LBFGS.step(closure)

    def predict(self, X_star):
        self.dnn.eval()
        x = torch.tensor(X_star[:, 0:1]).float().to(device)
        t = torch.tensor(X_star[:, 1:2]).float().to(device)
        with torch.no_grad():
            u = self.net_u(x, t)
        return u.cpu().numpy()

# ==========================================
# 3. 数据生成与模型训练
# ==========================================
def main():
    nu = 0.01 / np.pi
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] # [输入层, 隐藏层*8, 输出层]
    
    # --- 域定义 ---
    N_u = 100     # 初始/边界条件采样点数
    N_f = 10000   # 域内物理配点数 (无网格方法的核心，可类比为内部“网格”)
    
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0
    
    # --- 1. 生成初始条件点 (t = 0) ---
    x_ic = np.random.uniform(x_min, x_max, (N_u, 1))
    t_ic = np.zeros_like(x_ic)
    X_ic = np.hstack((x_ic, t_ic))
    u_ic = -np.sin(np.pi * x_ic)
    
    # --- 2. 生成边界条件点 (x = -1 和 x = 1) ---
    t_bc1 = np.random.uniform(t_min, t_max, (N_u // 2, 1))
    x_bc1 = np.full_like(t_bc1, x_min)
    t_bc2 = np.random.uniform(t_min, t_max, (N_u // 2, 1))
    x_bc2 = np.full_like(t_bc2, x_max)
    
    X_bc = np.vstack((np.hstack((x_bc1, t_bc1)), np.hstack((x_bc2, t_bc2))))
    u_bc = np.zeros((N_u, 1))
    
    # --- 3. 生成 PDE 配点 (时空域内随机拉丁超立方抽样或均匀随机) ---
    x_f = np.random.uniform(x_min, x_max, (N_f, 1))
    t_f = np.random.uniform(t_min, t_max, (N_f, 1))
    X_f = np.hstack((x_f, t_f))
    
    # 结合 BC 和 IC 的点作为域内配点的一部分
    X_f = np.vstack((X_f, X_ic, X_bc))
    
    # --- 实例化并训练模型 ---
    model = PhysicsInformedNN(X_f, X_ic, u_ic, X_bc, u_bc, layers, nu)
    
    # PINN 标准训练范式：先用 Adam 避开平坦区域快速下降，再用 L-BFGS 精确寻优
    model.train_adam(epochs=3000)
    model.train_lbfgs()
    
    # --- 后处理可视化 ---
    x_star = np.linspace(x_min, x_max, 256)
    t_star = np.linspace(t_min, t_max, 100)
    X, T = np.meshgrid(x_star, t_star)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_pred = model.predict(X_star)
    U_pred = u_pred.reshape(T.shape)
    
    plt.figure(figsize=(8, 4))
    plt.pcolor(T, X, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted Velocity field u(x, t)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()