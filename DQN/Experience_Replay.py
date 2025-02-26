import torch
import math
import numpy as np

class SumTreeGPU:
    def __init__(self, capacity, priorities=None, device = 'cuda'):
        """
        初始化 SumTree，并支持批量初始化所有优先级
        :param capacity: 叶节点数量，必须是 2 的幂次方
        :param priorities: 初始优先级张量，可选，大小为 capacity
        """

        self.device = device

        with torch.no_grad():
            assert (capacity > 0 and math.log2(capacity).is_integer()) == 1, "容量需要为2的正整数次方"

            self.capacity = capacity
            self.tree = torch.zeros(2 * capacity - 1, device=self.device, dtype=torch.float32)  # 初始化树
            # print(self.tree)
            self.index = 0  # 当前存储位置

            if priorities is not None:
                assert len(priorities) == capacity, "优先级张量的大小必须等于容量"
                self.init_with_priorities(priorities)

    def init_with_priorities(self, priorities):
        """
        使用初始优先级批量初始化 SumTree
        :param priorities: 初始优先级张量，大小为 capacity
        """
        with torch.no_grad():
            leaf_indices = torch.arange(self.capacity, device=self.device, dtype=torch.int32) + self.capacity - 1
            # print(leaf_indices)
            self.tree[leaf_indices] = priorities  # 初始化叶节点
            # print(self.tree)
            self.update_all_from_leaves()  # 逐层更新父节点

    def update_all_priorities(self, priorities):
        """
        一次性更新所有节点的优先级
        :param priorities: 新的优先级张量，大小为 capacity
        """
        with torch.no_grad():
            assert len(priorities) == self.capacity, "优先级张量的大小必须等于容量"
            leaf_indices = torch.arange(self.capacity, device=self.device, dtype=torch.int32) + self.capacity - 1
            self.tree[leaf_indices] = priorities  # 更新叶节点优先级
            self.update_all_from_leaves()  # 逐层更新父节点

    def update_all_from_leaves(self):
        """
        从叶节点开始批量更新整棵树
        """
        with torch.no_grad():
            num_nodes = self.capacity
            while num_nodes > 1:
                num_nodes //= 2
                # print(num_nodes)
                parent_indices = torch.arange(num_nodes, device=self.device, dtype=torch.int32) - 1 + num_nodes

                left_children = 2 * parent_indices + 1
                right_children = left_children + 1

                # 计算父节点的值
                self.tree[parent_indices] = self.tree[left_children] + self.tree[right_children]
                # print(parent_indices)
                # print(left_children)
                # print(right_children)

    def total_priority(self):
        """
        获取总优先级（根节点的值）
        """
        with torch.no_grad():
            return self.tree[0].item()

    def get_leaf_node(self, value):
        with torch.no_grad():
            # 根据给定的值 value 通过二叉树查找一个叶节点
            node_index = 0  # 从根节点开始
            while node_index < self.capacity - 1:  # 遇到叶节点时停止
                left = 2 * node_index + 1
                right = 2 * node_index + 2
                if self.tree[left] >= value:
                    node_index = left
                else:
                    value -= self.tree[left]
                    node_index = right
            return node_index - (self.capacity - 1), self.tree[node_index].item()  # 返回叶节点的实际数据索引

    def print_tree(self):
        print('prioritized_SumTree:\n',self.tree)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, agent_pos_dim, agent_action_dim, agent_num, tasks_num, n_step_parameter=10, alpha=0.6, beta=0.4, gamma=0.99, epsilon=1e-12, device='cuda'):
        """
        初始化优先经验回放
        :param capacity: 经验池大小
        :param alpha: 优先级的比例系数
        """

        self.device = device

        with torch.no_grad():
            self.capacity = capacity

            self.trajectory_reward_tensor = torch.zeros(capacity, 1, device=self.device, dtype=torch.float32)
            self.agent_pos_n1_tensor = torch.zeros(capacity, (agent_pos_dim * agent_num), device=self.device, dtype=torch.float32)
            self.agent_pos_1_tensor = torch.zeros(capacity, (agent_pos_dim * agent_num), device=self.device, dtype=torch.float32)
            self.agent_action_1_tensor = torch.zeros(capacity, (agent_action_dim * agent_num), device=self.device, dtype=torch.float32)
            self.TD_error_tensor = torch.zeros(capacity, 1, device=self.device, dtype=torch.float32)
            self.TD_target_tensor = torch.zeros(capacity, 1, device=self.device, dtype=torch.float32)
            self.tasks_state_1_tensor = torch.zeros(capacity, tasks_num, device=self.device, dtype=torch.float32)
            self.tasks_state_n1_tensor = torch.zeros(capacity, tasks_num, device=self.device, dtype=torch.float32)

            self.sum_tree = SumTreeGPU(capacity, device=device)
            self.sum_tree.init_with_priorities(
                torch.full((capacity,), 1 / capacity, device=self.device, dtype=torch.float32))

            self.pow_gamma_n = pow(gamma, n_step_parameter)

            self.epsilon = epsilon
            self.alpha = alpha
            self.beta = beta

    def load_from_file(self,path):
        with torch.no_grad():
            # print(path)
            loaded_tensors = torch.load(path, weights_only=True, map_location=self.device)
            self.trajectory_reward_tensor = loaded_tensors['trajectory_reward_tensor'].float()
            self.agent_pos_n1_tensor = loaded_tensors['agent_pos_n1_tensor'].float()
            self.agent_pos_1_tensor = loaded_tensors['agent_pos_1_tensor'].float()
            self.agent_action_1_tensor = loaded_tensors['agent_action_1_tensor'].float()
            self.tasks_state_1_tensor = loaded_tensors['tasks_state_1_tensor'].float()
            self.tasks_state_n1_tensor = loaded_tensors['tasks_state_n1_tensor'].float()
            # self.set_priorities(loaded_tensors['priorities_tensor'])

    def update_priorities(self, q_sn1_usn1_tensor, q_s1_a1_tensor):
        with torch.no_grad():
            self.TD_target_tensor = self.trajectory_reward_tensor + self.pow_gamma_n * q_sn1_usn1_tensor
            self.TD_error_tensor =  self.TD_target_tensor - q_s1_a1_tensor
            absolute_TD_error_tensor = torch.abs(self.TD_error_tensor)
            total_priority = torch.sum(absolute_TD_error_tensor ** self.alpha)
            normalized_priorities = (absolute_TD_error_tensor ** self.alpha) / total_priority + self.epsilon
            normalized_priorities = normalized_priorities.squeeze(1).float()
            # print(normalized_priorities.shape)
            # print(len(normalized_priorities))
            self.sum_tree.update_all_priorities(normalized_priorities)

    def sample(self, batch_size):
        """
        从经验池中采样
        :param batch_size: 采样的批量大小
        :param beta: 重要性采样权重的修正系数
        """
        """基于优先级进行采样"""
        with torch.no_grad():
            indices = []
            weights = []

            # 每次从根节点进行采样
            for _ in range(batch_size):
                # 从总优先级范围内采样一个随机值
                s = np.random.uniform(0, 1)
                index, priority = self.sum_tree.get_leaf_node(s)
                indices.append(index)
                # 计算重要性采样权重
                weight = (self.capacity * priority) ** (-self.beta)
                weights.append(weight)

            # print(weights)
            weights_np_array = np.array(weights)
            # print(weights_np_array)
            weights_tensor = torch.tensor(weights_np_array).unsqueeze(1).float().to(self.device)
            # print(weights_tensor)
            # print(indices)
            # print(self.s1_a1_tensor)
            # 返回采样的权重和数据
            '''return (
            self.agent_pos_1_tensor[indices], self.agent_action_1_tensor[indices], self.agent_pos_n1_tensor[indices],
            self.trajectory_reward_tensor[indices],
            weights_tensor,
            self.tasks_state_1_tensor[indices], self.tasks_state_n1_tensor[indices])'''
            return (
                self.agent_pos_1_tensor[indices], self.tasks_state_1_tensor[indices], self.agent_action_1_tensor[indices],
                self.TD_target_tensor[indices], weights_tensor
            )

    def change_alpha(self,new_alpha):
        self.alpha = new_alpha

    def change_beta(self, new_beta):
        self.beta = new_beta

    def print_buffer_info(self):
        print('capacity:',self.capacity)
        print('alpha:',self.alpha)
        print('beta:',self.beta)

    '''def print_buffer_data(self):
        print('trajectory_reward_tensor:\n',self.trajectory_reward_tensor)
        print('sn1_tensor:\n',self.sn1_tensor)
        print('s1_a1_tensor:\n',self.s1_a1_tensor)
        self.sum_tree.print_tree()'''

    def adjust_alpha_beta_sigmoid(self ,t, T, alpha_max, alpha_min, beta_max, beta_min, k=20, p_m=0.5):
        p = t / T  # Normalize step
        alpha = alpha_max - (alpha_max - alpha_min) / (1 + np.exp(-k * (p - p_m)))
        beta = beta_min + (beta_max - beta_min) / (1 + np.exp(-k * (p - p_m)))
        self.change_alpha(alpha)
        self.change_beta(beta)
        print("alpha,beta:",alpha,beta)

    def return_loss_mean(self):
        return (0.5 * (self.TD_error_tensor**2)).mean()

    def sample_uniform(self, mini_batch_size):
        indices = torch.randperm(self.agent_pos_1_tensor.size(0))[:mini_batch_size]
        return (
            self.agent_pos_1_tensor[indices], self.tasks_state_1_tensor[indices], self.agent_action_1_tensor[indices],
            self.TD_target_tensor[indices], None
        )

    def update_TD_info(self, q_sn1_usn1_tensor, q_s1_a1_tensor):
        self.TD_target_tensor = self.trajectory_reward_tensor + self.pow_gamma_n * q_sn1_usn1_tensor
        self.TD_error_tensor = self.TD_target_tensor - q_s1_a1_tensor