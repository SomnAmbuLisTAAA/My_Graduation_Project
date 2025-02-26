import torch
import torch.nn as nn

class CentralCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CentralCritic, self).__init__()

        size = 128

        self.fc1 = nn.Linear(state_dim + action_dim, size)
        self.prelu1 = nn.PReLU(num_parameters=size)  # PReLU 1

        self.fc2 = nn.Linear(size, size)
        self.prelu2 = nn.PReLU(num_parameters=size)  # PReLU 2

        self.fc3 = nn.Linear(size, size)
        self.prelu3 = nn.PReLU(num_parameters=size)  # PReLU 3

        self.fc4 = nn.Linear(size, size)
        self.prelu4 = nn.PReLU(num_parameters=size)  # PReLU 4

        # 输出层
        self.q1_out = nn.Linear(size, 1)  # 对应 actor1
        self.q2_out = nn.Linear(size, 1)  # 对应 actor2

    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))

        x1 = self.prelu3(self.fc3(x))
        x2 = self.prelu4(self.fc4(x))

        q1 = self.q1_out(x1)  # Q1 只和 actor1 相关
        q2 = self.q2_out(x2)  # Q2 只和 actor2 相关
        return q1, q2