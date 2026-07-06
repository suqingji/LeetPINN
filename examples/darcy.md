# 二维定常达西问题 (Darcy 2D) 的 PINN 求解

## 一、 问题描述

达西方程（Darcy equation）是一个描述流体在多孔介质中低速流动时渗流规律的二阶椭圆型偏微分方程，被广泛应用于水利工程、石油工程等领域。通常采用有限元法（FEM）等数值方法进行求解，但基于物理信息的神经网络（PINNs）能以无网格（Mesh-free）的方式逼近传统数值方法的求解精度。

考虑二维正方体 $\Omega=(0, 1)\times(0, 1)$，该正方体的边界为 $\Gamma$。在忽略重力影响的 $\Omega$ 范围内，流体压力 $p$ 和速度场 $u$（包含 $x, y$ 两个方向分量，此处分别记为 $u, v$）满足定常 2D Darcy 方程：

$$u + \frac{\partial p}{\partial x} = 0$$

$$v + \frac{\partial p}{\partial y} = 0$$

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = f$$

本案例使用迪利克雷（Dirichlet）边界条件，形式如下：

$$u = -2 \pi \cos(2 \pi x) \cos(2 \pi y) \quad (x, y)\in\Gamma$$

$$v = 2 \pi \sin(2 \pi x) \sin(2 \pi y) \quad (x, y)\in\Gamma$$

$$p = \sin(2 \pi x) \cos(2 \pi y) \quad (x, y)\in\Gamma$$

我们要利用 PINN 学习 forcing function $f = 8 \pi^2 \sin(2 \pi x)\cos(2 \pi y)$ 时，位置坐标到相应物理量的映射：$(x, y) \mapsto (u, v, p)$。

## 二、 求解流程与技术路线

本 PyTorch 实现的流程与原 MindSpore 文件保持一致：

1. **创建数据集 (Data Generation)**：使用均匀随机分布生成域内配点（用于 PDE 残差约束）和边界配点（用于边界条件约束）。
2. **构建模型 (Model Definition)**：构建输入通道为 2（坐标 $x, y$），输出通道为 3（物理量 $u, v, p$）的神经网络。网络采用 128 个神经元，包含基于 Tanh 激活函数的残差块（Residual Block）以防梯度消失。
3. **优化器 (Optimizer)**：采用 Adam 优化算法。
4. **2D Darcy 损失函数 (PDE & BC Loss)**：构建由连续性方程与动量方程推导出的 `loss_1`、`loss_2`、`loss_3` 以及对应的边界残差。
5. **模型训练与可视化**：多 Epoch 迭代寻优，最后通过 Matplotlib 对比精准的解析解与 PINN 预测压力场。