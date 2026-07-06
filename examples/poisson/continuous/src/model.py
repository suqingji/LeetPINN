import math
import torch
import torch.nn as nn

class ExpDeacy(nn.Module):
    """Exponentially deacy activation function."""
    def __init__(self):
        super(ExpDeacy, self).__init__()
        self.factor = -1.0 / (2 * math.e)

    def forward(self, x):
        return x * torch.exp(self.factor * (x ** 2))

class LinearWN(nn.Module):
    """A linear layer with weight normalization."""
    def __init__(self, in_features, out_features):
        super(LinearWN, self).__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features))
        # 对应 HeUniform 和 Normal(sigma=.1) 初始化
        nn.init.kaiming_uniform_(self.linear.weight_v, a=math.sqrt(5))
        if self.linear.bias is not None:
            nn.init.normal_(self.linear.bias, std=0.1)

    def forward(self, x):
        return self.linear(x)

def create_model(input_size, base_neurons):
    """Create a network."""
    return nn.Sequential(
        nn.Linear(input_size, 8 * base_neurons),
        ExpDeacy(),
        nn.Linear(8 * base_neurons, 4 * base_neurons),
        ExpDeacy(),
        nn.Linear(4 * base_neurons, 2 * base_neurons),
        ExpDeacy(),
        nn.Linear(2 * base_neurons, base_neurons),
        ExpDeacy(),
        LinearWN(base_neurons, 1)
    )