# 二维圆柱绕流 (Navier-Stokes 2D) 的 PINN 求解

## 一、 问题描述

圆柱绕流是流体力学中的经典样板问题。在低雷诺数（如本例中的 `Re = 100`）下，流体会表现出特殊的粘性与惯性相互作用现象。由于传统的数值方法（如有限元 FEM、有限差分 FDM）需要极其精细的网格离散化，计算成本高昂，我们采用物理信息神经网络（PINNs）来进行数据驱动的快速求解。

控制流体运动的核心是不可压缩的纳维-斯托克斯（Navier-Stokes）方程。其二维无量纲形式如下：

**1. 连续性方程 (Mass Conservation):**

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**2. 动量方程 (Momentum Conservation):**

$$\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} \left(\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}\right)$$

$$\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} \left(\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2}\right)$$

本案例利用神经网络学习**时空坐标**到**流场物理量**的连续映射：

$$(x, y, t) \mapsto (u, v, p)$$

## 二、 求解流程与技术路线

对标 MindSpore 的流程，我们的 PyTorch 实现分为以下几个核心步骤：

1. **创建数据集**：构建 PDE 内部配点、初始条件（IC）和边界条件（BC）的时空采样点。
2. **构建模型**：建立输入层为3通道（$x, y, t$），输出层为3通道（$u, v, p$）的全连接神经网络，激活函数使用 `Tanh`。
3. **NavierStokes2D 物理约束**：利用 PyTorch 的 `autograd` 自动微分引擎，计算速度场的一阶和二阶导数，并构造残差损失。
4. **自适应损失的多任务学习**：由于包含 PDE、IC、BC 多个 Loss，传统固定权重极易导致偏科。引入 Kendall 等人提出的不确定性权重算法（Uncertainty Weighting），将各项损失的权重设为可学习参数，在训练中动态自适应调节。
5. **模型训练**：使用 Adam 优化器进行端到端求解。