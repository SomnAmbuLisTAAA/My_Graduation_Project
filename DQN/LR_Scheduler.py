from collections import deque

class RecentLossLRScheduler:
    def __init__(self, basic_lr, factor=0.1, patience=5, window_size=5, threshold=0.01, min_lr=1e-6 ):

        self.basic_lr = basic_lr
        self.min_lr = min_lr
        self.factor = factor  # 学习率调整因子
        self.patience = patience  # 连续多少次没有明显变化就调整学习率
        self.window_size = window_size  # 计算最近损失的窗口大小
        self.threshold = threshold  # 相对变化阈值
        self.loss_history = deque(maxlen=window_size)  # 使用队列记录最近的损失
        self.best_loss = float('inf')  # 最优损失初始化为正无穷
        self.count = 0

    def step(self, loss):

        if self.best_loss != float('inf'):
            relative_change = (self.best_loss - loss) / self.best_loss
        else:
            relative_change = float('inf')

        if relative_change < self.threshold:
            self.count += 1
        else:
            self.count = 0

        if self.count >= self.patience:
            self.count = 0
            self.basic_lr = self.basic_lr * self.factor

        self.loss_history.append(loss) # 记录当前损失
        self.best_loss = min(self.loss_history)

        if self.basic_lr <= self.min_lr:
            self.basic_lr = self.min_lr

        return self.basic_lr

    def clear_memory(self):
        self.best_loss = float('inf')  # 最优损失初始化为正无穷
        self.loss_history.clear()
        self.count = 0