import torch
import torch.nn as nn

class SequentialMultiLayerNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        多隐藏层神经网络，含批归一化
        Attributes:
            input_size(int): 输入维数
            hidden_size(int): 单隐藏层神经元个数
            num_layers(int): 隐藏层层数
            output_size(int): 输出维数
        """
        super(SequentialMultiLayerNN, self).__init__()
        layers = []

        # 输入到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))   # 输入层连接到第一个隐藏层
        # layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
        layers.append(nn.ReLU())    # 使用ReLU激活函数

        # 中间隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 连接到下一个隐藏层
            # layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        # 使用 nn.Sequential 包装
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)