import torch
import torch.nn as nn

# SequentialMultiLayerNN: 多隐藏层神经网络，含批归一化
# input_size: 输入维数
# hidden_size: 单隐藏层神经元个数
# num_layers: 隐藏层层数
# output_size: 输出维数

class SequentialMultiLayerNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SequentialMultiLayerNN, self).__init__()
        layers = []

        # 输入到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))   # 输入层连接到第一个隐藏层
        layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
        layers.append(nn.ReLU())    # 使用ReLU激活函数

        # 中间隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 连接到下一个隐藏层
            layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        # 使用 nn.Sequential 包装
        self.network = nn.Sequential(*layers)
        # print(layers)

    def forward(self, x):
        return self.network(x)



class ScaledTanh(nn.Module):
    def __init__(self, scale=10, factor=0.1):
        super(ScaledTanh, self).__init__()
        self.scale = scale  # 输出范围
        self.factor = factor  # 缩放输入

    def forward(self, x):
        return self.scale * torch.tanh(self.factor * x)



class SequentialMultiLayerNN_with_scaled_tanh(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, scale, factor):
        super(SequentialMultiLayerNN_with_scaled_tanh, self).__init__()
        layers = []

        # 输入到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))   # 输入层连接到第一个隐藏层
        layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
        layers.append(nn.ReLU())    # 使用ReLU激活函数

        # 中间隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 连接到下一个隐藏层
            layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))

        # 使用 nn.Sequential 包装
        self.network = nn.Sequential(*layers)
        # print(layers)

        # 激活函数参数
        self.scale = scale
        self.factor = factor

    def forward(self, x):
        # 获取网络输出
        x = self.network(x)
        # 应用自定义激活函数
        x = self.scale * torch.tanh(self.factor * x)
        return x

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DeepLearning.NN_Init import SequentialMultiLayerNN

# 参数调整
input_size = 784       # 输入特征大小
hidden_size = 128      # 隐藏层神经元数量
num_layers = 3         # 隐藏层数量
output_size = 2        # 输出维数
learning_rate = 0.001  # 学习率
batch_size = 64        # Mini-Batch 大小
epochs = 1           # 训练轮数
weight_decay = 0.01    # L2 正则化强度（权重衰减因子）
patience = 3           # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5           # 学习率衰减参数

# 随机生成数据
x = torch.randn(2048, input_size)  # 随机生成输入特征
y = torch.randint(0, 2, (2048, 2)).float()  # 随机生成 0 或 1 标签

if __name__ == '__main__':  # 在脚本的主入口处添加此保护代码来确保多进程启动时不会重复导入主模块
    # 数据包装为 DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 初始化模型
    model = SequentialMultiLayerNN(input_size, hidden_size, num_layers, output_size)

    # 检查是否有 GPU 可用，如果有则使用 GPU 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # device = 'cpu'
    model = model.to(device)

    # 损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 使用 Adam 优化器，并添加 L2 正则化（通过 weight_decay 参数）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 使用 ReduceLROnPlateau 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience , factor=factor)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):

            # 将输入和标签移到 GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            logistics = model(batch_x)  # 输出 logistics（未激活值）
            loss = criterion(logistics, batch_y)
            print(loss)

            # 清零梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 打印训练信息
            if (batch_idx + 1) % 16 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 在每个 epoch 结束时更新学习率
        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)  # 通过平均损失来调整学习率

    # 测试模型
    with torch.no_grad():
        test_x = torch.randn(5, input_size).to(device)  # 随机测试数据并移到 GPU
        logistics = model(test_x)
        predictions = torch.sigmoid(logistics) > 0.5  # 通过 sigmoid 转为二分类预测
        print("Test Predictions:", predictions)
'''