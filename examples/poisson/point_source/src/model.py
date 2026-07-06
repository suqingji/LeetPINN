import torch
import torch.nn as nn

class SinAct(nn.Module):
    """Sine activation function."""
    def forward(self, x):
        return torch.sin(x)

class FCResidualBlock(nn.Module):
    def __init__(self, neurons, act):
        super(FCResidualBlock, self).__init__()
        self.linear = nn.Linear(neurons, neurons)
        self.act = act

    def forward(self, x):
        return self.act(self.linear(x)) + x  # 带有残差连接

class MultiScaleFCSequential(nn.Module):
    def __init__(self, in_channels, out_channels, layers, neurons, residual=True, act="sin",
                 num_scales=2, scale_factor=2.0, input_scale=[10., 10.]):
        super(MultiScaleFCSequential, self).__init__()
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.input_scale = torch.tensor(input_scale, dtype=torch.float32)
        
        activation = SinAct() if act == "sin" else nn.Tanh()
        
        self.networks = nn.ModuleList()
        for i in range(num_scales):
            net = []
            net.append(nn.Linear(in_channels, neurons))
            net.append(activation)
            for _ in range(layers - 2):
                if residual:
                    net.append(FCResidualBlock(neurons, activation))
                else:
                    net.append(nn.Linear(neurons, neurons))
                    net.append(activation)
            net.append(nn.Linear(neurons, out_channels))
            self.networks.append(nn.Sequential(*net))

    def forward(self, x):
        device = x.device
        self.input_scale = self.input_scale.to(device)
        
        out = 0
        for i in range(self.num_scales):
            # 对输入进行多尺度伸缩
            scale = self.input_scale * (self.scale_factor ** i)
            x_scaled = x * scale
            out = out + self.networks[i](x_scaled)
        return out