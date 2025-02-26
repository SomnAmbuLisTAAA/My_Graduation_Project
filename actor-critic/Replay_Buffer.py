import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque





class ReplayBuffer(Dataset):
    def __init__(self, max_size=1_000_000, load_path=None):
        """
        经验回放池，支持多种数据类型存储。

        :param max_size: 经验池最大容量
        :param load_path: 可选，从文件加载数据
        """
        self.max_size = max_size
        self.data = {
            "trajectory_reward": np.zeros((max_size, 2), dtype=np.float32),
            "o1_1": np.zeros((max_size, 3), dtype=np.float32),
            "o2_1": np.zeros((max_size, 3), dtype=np.float32),
            "o1_n1": np.zeros((max_size, 3), dtype=np.float32),
            "o2_n1": np.zeros((max_size, 3), dtype=np.float32),
            "state_1": np.zeros((max_size, 9), dtype=np.float32),
            "state_1_without_actions": np.zeros((max_size, 5), dtype=np.float32),
            "state_n1_without_actions": np.zeros((max_size, 5), dtype=np.float32)
        }
        self.index = 0
        self.full = False

        if load_path:
            self.load_data(load_path)

    def update_data(self, trajectory_rewards, o1_1, o2_1, o1_n1, o2_n1, state_1, state_1_without_actions,
                    state_n1_without_actions, shuffle=False):
        """
        添加新数据，并自动删除最旧数据（FIFO 方式）。如果 `shuffle` 为 True，将数据打乱后再存储。
        """
        num_new = trajectory_rewards.shape[0]

        if shuffle:
            # 在存储之前手动打乱数据
            indices = np.random.permutation(num_new)
            trajectory_rewards = trajectory_rewards[indices]
            o1_1 = o1_1[indices]
            o2_1 = o2_1[indices]
            o1_n1 = o1_n1[indices]
            o2_n1 = o2_n1[indices]
            state_1 = state_1[indices]
            state_1_without_actions = state_1_without_actions[indices]
            state_n1_without_actions = state_n1_without_actions[indices]

        if num_new >= self.max_size:
            # 如果新数据太多，直接覆盖整个 buffer
            self.data["trajectory_reward"][:] = trajectory_rewards[-self.max_size:]
            self.data["o1_1"][:] = o1_1[-self.max_size:]
            self.data["o2_1"][:] = o2_1[-self.max_size:]
            self.data["o1_n1"][:] = o1_n1[-self.max_size:]
            self.data["o2_n1"][:] = o2_n1[-self.max_size:]
            self.data["state_1"][:] = state_1[-self.max_size:]
            self.data["state_1_without_actions"][:] = state_1_without_actions[-self.max_size:]
            self.data["state_n1_without_actions"][:] = state_n1_without_actions[-self.max_size:]
            self.index = 0
            self.full = True
        else:
            end_index = self.index + num_new
            if end_index <= self.max_size:
                self.data["trajectory_reward"][self.index:end_index] = trajectory_rewards
                self.data["o1_1"][self.index:end_index] = o1_1
                self.data["o2_1"][self.index:end_index] = o2_1
                self.data["o1_n1"][self.index:end_index] = o1_n1
                self.data["o2_n1"][self.index:end_index] = o2_n1
                self.data["state_1"][self.index:end_index] = state_1
                self.data["state_1_without_actions"][self.index:end_index] = state_1_without_actions
                self.data["state_n1_without_actions"][self.index:end_index] = state_n1_without_actions
            else:
                first_part = self.max_size - self.index
                self.data["trajectory_reward"][self.index:] = trajectory_rewards[:first_part]
                self.data["o1_1"][self.index:] = o1_1[:first_part]
                self.data["o2_1"][self.index:] = o2_1[:first_part]
                self.data["o1_n1"][self.index:] = o1_n1[:first_part]
                self.data["o2_n1"][self.index:] = o2_n1[:first_part]
                self.data["state_1"][self.index:] = state_1[:first_part]
                self.data["state_1_without_actions"][self.index:] = state_1_without_actions[:first_part]
                self.data["state_n1_without_actions"][self.index:] = state_n1_without_actions[:first_part]

                self.data["trajectory_reward"][:end_index - self.max_size] = trajectory_rewards[first_part:]
                self.data["o1_1"][:end_index - self.max_size] = o1_1[first_part:]
                self.data["o2_1"][:end_index - self.max_size] = o2_1[first_part:]
                self.data["o1_n1"][:end_index - self.max_size] = o1_n1[first_part:]
                self.data["o2_n1"][:end_index - self.max_size] = o2_n1[first_part:]
                self.data["state_1"][:end_index - self.max_size] = state_1[first_part:]
                self.data["state_1_without_actions"][:end_index - self.max_size] = state_1_without_actions[first_part:]
                self.data["state_n1_without_actions"][:end_index - self.max_size] = state_n1_without_actions[first_part:]

            self.index = (self.index + num_new) % self.max_size
            if self.index == 0:
                self.full = True

    def load_data(self, file_path):
        """
        从 npz/csv 文件加载数据（如果数据量超出 buffer 大小，仅保留最新的数据）。
        :param file_path: 数据文件路径
        """
        if file_path.endswith('.npz'):
            loaded_data = np.load(file_path)
            trajectory_rewards = loaded_data['trajectory_reward']
            o1_1 = loaded_data['o1_1']
            o2_1 = loaded_data['o2_1']
            o1_n1 = loaded_data['o1_n1']
            o2_n1 = loaded_data['o2_n1']
            state_1 = loaded_data['state_1']
            state_1_without_actions = loaded_data['state_1_without_actions']
            state_n1_without_actions = loaded_data['state_n1_without_actions']
        elif file_path.endswith('.csv'):
            loaded_data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
            trajectory_rewards = loaded_data[:, :2]
            o1_1 = loaded_data[:, 2:9]
            o2_1 = loaded_data[:, 9:16]
            o1_n1 = loaded_data[:, 16:23]
            o2_n1 = loaded_data[:, 23:30]
            state_1 = loaded_data[:, 30:39]
            state_1_without_actions = loaded_data[:, 39:44]
            state_n1_without_actions = loaded_data[:, 44:49]
        else:
            raise ValueError("只支持 .npz 或 .csv 格式的文件！")

        print(f"从 {file_path} 加载数据")

        # 仅保留最新的数据，防止超出 buffer 容量
        self.update_data(trajectory_rewards, o1_1, o2_1, o1_n1, o2_n1, state_1, state_1_without_actions,
                         state_n1_without_actions)

    def save_data(self, file_path):
        """
        保存当前经验到 .npz 文件
        :param file_path: 目标文件路径
        """
        np.savez(file_path,
                 trajectory_reward=self.data["trajectory_reward"][:self.__len__()],
                 o1_1=self.data["o1_1"][:self.__len__()],
                 o2_1=self.data["o2_1"][:self.__len__()],
                 o1_n1=self.data["o1_n1"][:self.__len__()],
                 o2_n1=self.data["o2_n1"][:self.__len__()],
                 state_1=self.data["state_1"][:self.__len__()],
                 state_1_without_actions=self.data["state_1_without_actions"][:self.__len__()],
                 state_n1_without_actions=self.data["state_n1_without_actions"][:self.__len__()])
        print(f"经验池已保存到 {file_path}")

    def shuffle_data(self):
        """手动打乱数据（适用于某些情况下手动 shuffle）"""
        indices = np.random.permutation(self.__len__())
        for key in self.data:
            self.data[key] = self.data[key][indices]

    def __len__(self):
        return self.max_size if self.full else self.index

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.data.items()}





class ReplayBufferDeque:
    def __init__(self, capacity):
        """
        初始化经验回放池
        :param capacity: 回放池容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 固定容量的deque

    def add_batch(self, trajectory_rewards, o1_1, o2_1, o1_n1, o2_n1,
                  state_1, state_1_without_actions, state_n1_without_actions):
        """
        批量将经验存入回放池
        :param trajectory_rewards: 轨迹奖励 (batch_size, 1)
        :param o1_1: o1_1数据 (batch_size, o_size_1)
        :param o2_1: o2_1数据 (batch_size, o_size_2)
        :param o1_n1: o1_n1数据 (batch_size, o_size_1)
        :param o2_n1: o2_n1数据 (batch_size, o_size_2)
        :param state_1: 当前状态1 (batch_size, state_size)
        :param state_1_without_actions: 当前状态1不含动作 (batch_size, state_size)
        :param state_n1_without_actions: 下一个状态n1不含动作 (batch_size, state_size)
        """
        batch_size = trajectory_rewards.size(0)
        for i in range(batch_size):
            experience = {
                "trajectory_reward": trajectory_rewards[i],
                "o1_1": o1_1[i],
                "o2_1": o2_1[i],
                "o1_n1": o1_n1[i],
                "o2_n1": o2_n1[i],
                "state_1": state_1[i],
                "state_1_without_actions": state_1_without_actions[i],
                "state_n1_without_actions": state_n1_without_actions[i]
            }
            self.buffer.append(experience)  # 超过容量会自动删除最旧的数据

    def sample(self, batch_size):
        """
        随机抽取一个batch
        :param batch_size: 批大小
        :return: 随机抽取的批量经验
        """
        if len(self.buffer) < batch_size:
            raise ValueError("回放池中的数据不足以提供一个完整的batch")
        batch = random.sample(self.buffer, batch_size)  # 随机抽取一个批次
        batch_dict = {key: torch.stack([experience[key] for experience in batch]) for key in
                      batch[0]}  # 将batch中的各个经验合并为tensor
        return batch_dict

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return f"ReplayBuffer(size={len(self.buffer)}, capacity={self.capacity})"

    def save_to_file(self, file_path):
        """
        将回放池的数据保存到文件
        :param file_path: 文件路径
        """
        # 将经验池转换为字典
        data = {
            "buffer": list(self.buffer),
            "capacity": self.capacity
        }
        torch.save(data, file_path)
        print(f"Data saved to {file_path}")

    def load_from_file(self, file_path):
        """
        从文件加载回放池数据
        :param file_path: 文件路径
        """
        data = torch.load(file_path)
        self.capacity = data["capacity"]
        self.buffer = deque(data["buffer"], maxlen=self.capacity)
        print(f"Data loaded from {file_path}")