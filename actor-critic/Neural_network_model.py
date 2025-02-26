import torch
import torch.nn as nn
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)

class SequentialMultiLayerNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, scale=1, factor=0.1):
        """
        多隐藏层神经网络
        每次更改actor网络的输出限幅时，必须调整tanh对应的两个参数以确保梯度不会过大或过小
        Attributes:
            input_size(int): 输入维数
            hidden_size(int): 单隐藏层神经元个数
            num_layers(int): 隐藏层层数
            output_size(int): 输出维数
        """
        super(SequentialMultiLayerNN, self).__init__()
        layers = []

        self.scale = scale
        self.factor = factor

        # 输入到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))   # 输入层连接到第一个隐藏层
        # layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
        layers.append(nn.PReLU(num_parameters=hidden_size))    # 使用ReLU激活函数

        # 中间隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 连接到下一个隐藏层
            # layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.PReLU(num_parameters=hidden_size))

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        # 使用 nn.Sequential 包装
        self.network = nn.Sequential(*layers)

        # 权重初始化
        # self._initialize_weights()

    def _initialize_weights(self):
        """ 自定义初始化方法 """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.zeros_(layer.bias)
            if isinstance(layer, nn.PReLU):
                init.constant_(layer.weight, 0.25)

    def forward(self, x):
        return self.network(x)

    def with_scaled_tanh(self, x):
        # 获取网络输出
        output = self.network(x)
        # 应用自定义激活函数
        tanh_output = self.scale * torch.tanh(self.factor * output)
        return tanh_output