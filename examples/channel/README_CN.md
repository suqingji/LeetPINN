# Channel Flow PINN 示例

这个示例使用 PyTorch 实现 PINN，求解二维稳态不可压 channel/Poiseuille 流动。

控制方程为稳态 Navier-Stokes 方程：

```text
u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0
u v_x + v v_y + p_y - nu (v_xx + v_yy) = 0
u_x + v_y = 0
```

计算区域为 `x in [0, L]`，`y in [-1, 1]`。默认边界条件：

- 入口 `x=0`：`u = umax * (1 - y^2)`，`v = 0`
- 上下壁 `y=±1`：无滑移，`u = 0`，`v = 0`
- 出口 `x=L`：`p = 0`

默认解析解为：

```text
u = umax * (1 - y^2)
v = 0
p = 2 * nu * umax * (L - x)
```

## 运行

在仓库根目录执行：

```bash
python3 examples/channel/train.py
```

如果想先快速检查流程，可以减少训练轮数和采样点：

```bash
python3 examples/channel/train.py --epochs 200 --n_f 1024 --n_wall 256 --n_inlet 128 --n_outlet 128
```

可选参数示例：

```bash
python3 examples/channel/train.py --device cuda --epochs 5000 --use_lbfgs
```

训练结束后会在 `outputs/channel` 下保存：

- `model.pth`：模型权重
- `loss.png`：训练损失曲线
- `velocity_u.png`：PINN 预测速度、解析速度和误差云图
