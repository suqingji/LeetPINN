## 一维Burgers

### 一、 问题描述 (Problem Formulation)

**1. 控制方程 (Governing Equation)**

一维黏性 Burgers 方程是流体力学中极其重要的非线性基础模型，它包含了 Navier-Stokes 方程中最核心的非线性对流项和耗散（扩散）项。其标准形式为：

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

其中，$x \in [-1, 1]$ 为空间坐标，$t \in [0, 1]$ 为时间，$u(x, t)$ 为流体速度，$\nu$ 为运动黏度。在本算例中，我们设定 $\nu = 0.01 / \pi$。

**2. 初始条件 (Initial Condition)**

$$u(x, 0) = -\sin(\pi x)$$

**3. 边界条件 (Boundary Condition)**

采用狄利克雷边界条件（Dirichlet BC）：

$$u(-1, t) = u(1, t) = 0$$

### 二、 PINN 求解框架 (Methodology)

与传统 CFD 离散控制方程并求解大型稀疏代数方程组不同，PINN 的核心思想是**用神经网络逼近连续物理场，并用自动微分（Automatic Differentiation）计算偏导数来约束物理定律。**

定义神经网络 $u_\theta(x, t)$，其输入为时空坐标 $(x, t)$，输出为速度 $u$。整个优化过程的损失函数（Loss Function）由三部分组成：

$$\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}$$

- **物理残差损失 ($\mathcal{L}_{PDE}$)**：在整个计算域内随机撒点（配点，Collocation points），强迫网络输出满足 Burgers 方程。

  $$\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left( \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} \right)^2$$

- **初始条件损失 ($\mathcal{L}_{IC}$)**：在 $t=0$ 处采样，约束初始状态。

- **边界条件损失 ($\mathcal{L}_{BC}$)**：在 $x=-1$ 和 $x=1$ 处采样，约束边界状态。