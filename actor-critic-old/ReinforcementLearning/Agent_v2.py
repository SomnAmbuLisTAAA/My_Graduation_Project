from math import gamma
import numpy as np
from collections import deque
from DeepLearning.NN_model import SequentialMultiLayerNN
import random
import ReinforcementLearning.Agent as agent
import torch
import torch.nn as nn
import torch.optim as optim



class Actor_Critic:
    class simplest_ac_agent:
        def __init__(self, observation_dim, action_dim, agent_radius, vx_min, vx_max, vy_min, vy_max, init_pos_x, init_pos_y, gamma=0.99):

            self.observation_dim = observation_dim
            self.action_dim = action_dim

            # 折扣因子
            self.gamma = gamma

            # 经验回放缓冲区
            self.action_memory = deque(maxlen=100)
            self.immediate_reward_memory = deque(maxlen=10)
            self.immediate_reward = 0.0
            self.trajectory_reward_memory = deque(maxlen=100)
            self.actor_loss_queue = deque(maxlen=10)
            self.critic_loss_queue = deque(maxlen=10)

            self.vx_max = abs(vx_max)
            self.vy_max = abs(vy_max)
            self.vx_min = vx_min
            self.vy_min = vy_min
            self.speed_limits = (self.vx_min, self.vx_max, self.vy_min, self.vy_max)

            self.agent_radius = agent_radius
            self.step = -1

            self.init_pos = np.array([init_pos_x,init_pos_y])
            self.pos = np.array([init_pos_x,init_pos_y])
            self.v = np.array([0,0])
            self.LastPos = np.array([init_pos_x,init_pos_y])
            self.reward = 0.0

            self.actor_network_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.critic_network_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.actor_network = SequentialMultiLayerNN(self.observation_dim,2,128, action_dim)
            self.critic_network = SequentialMultiLayerNN(self.observation_dim + action_dim,2,128, 1)

            self.actor_loss_info = 0.0
            self.critic_loss_info = 0.0

            self.learning_rate = 0.0004  # 学习率
            self.batch_size = 64  # Mini-Batch 大小
            self.epochs = 100  # 训练轮数
            self.weight_decay = 0.01  # L2 正则化强度（权重衰减因子）
            self.patience = 3  # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
            self.factor = 0.5  # 学习率衰减参数
            self.TD_error_weight = 0.8

            self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
            self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.actor_optimizer,
                                                                               mode='min', patience=self.patience,
                                                                               factor=self.factor)
            self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
            self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer,
                                                                              mode='min', patience=self.patience,
                                                                              factor=self.factor)

            self.critic_network = self.critic_network.to(self.critic_network_device)
            self.actor_network = self.actor_network.to(self.actor_network_device)

        def agent_init(self):
            self.v = np.array([0, 0])
            self.LastPos = self.init_pos
            self.pos = self.init_pos
            self.reward = 0.0
            self.immediate_reward = 0.0
            self.step = -1
            self.actor_loss_info = 0.0
            self.critic_loss_info = 0.0
            self.action_memory.clear()
            self.immediate_reward_memory.clear()
            self.trajectory_reward_memory.clear()
            self.actor_loss_queue.clear()
            self.critic_loss_queue.clear()

        def agent_random_init(self,width,height):
            self.v = np.array([0, 0])
            self.init_pos = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
            self.LastPos = self.init_pos
            self.pos = self.init_pos
            self.reward = 0.0
            self.immediate_reward = 0.0
            self.step = -1
            self.actor_loss_info = 0.0
            self.critic_loss_info = 0.0
            self.action_memory.clear()
            self.immediate_reward_memory.clear()
            self.trajectory_reward_memory.clear()
            self.actor_loss_queue.clear()
            self.critic_loss_queue.clear()

        def get_agent_pos(self):
            return self.pos

        def get_agent_dim(self):
            return self.observation_dim, self.action_dim

        def explore_act(self):
            self.v = np.array([random.uniform(self.vx_min, self.vx_max), random.uniform(self.vy_min, self.vy_max)])

        def policy_act(self,state):
            self.actor_network.eval()
            st = state.to(self.actor_network_device)
            at = self.actor_network(st)

            at = at.detach().cpu().numpy().flatten()
            self.v = np.clip(at, [self.vx_min, self.vy_min], [self.vx_max, self.vy_max])
            return self.v

        def update_pos(self,sample_time):
            self.step += 1
            self.LastPos = self.pos
            self.pos = self.pos + sample_time * self.v

        def get_agent_radius(self):
            return self.agent_radius

        def undo_pos(self):
            self.pos = self.LastPos

        def get_reward(self):
            return self.reward

        def get_last_pos(self):
            return self.LastPos

        def get_speed_limits(self):
            return self.speed_limits

        def save_model(self,actor_network_path,critic_network_path):
            torch.save(self.actor_network.state_dict(), actor_network_path)
            torch.save(self.critic_network.state_dict(), critic_network_path)

        def load_model(self,actor_network_path,critic_network_path):
            self.actor_network.load_state_dict(torch.load(actor_network_path, weights_only=True))
            self.critic_network.load_state_dict(torch.load(critic_network_path, weights_only=True))
            self.actor_network.eval()
            self.critic_network.eval()

        def save_experience(self,reward):
            self.immediate_reward = reward
            self.reward += self.immediate_reward * pow(self.gamma,self.step)
            self.action_memory.append(self.v)
            self.immediate_reward_memory.append(self.immediate_reward)
            if len(self.immediate_reward_memory) == self.immediate_reward_memory.maxlen:
                trajectory_reward = 0.0
                for i in range(len(self.immediate_reward_memory)):
                    trajectory_reward += self.immediate_reward_memory[i] * pow(self.gamma, i)
                    self.trajectory_reward_memory.append(trajectory_reward)

        def upgrade_network(self,s1_batch,sn1_batch):

            if len(self.trajectory_reward_memory) == self.trajectory_reward_memory.maxlen:

                self.actor_network.train()
                self.critic_network.train()

                sn1_batch = sn1_batch.to(self.actor_network_device)
                an1_batch = self.actor_network(sn1_batch)

                sn1_an1_batch = torch.cat((sn1_batch, an1_batch), dim=1)
                sn1_an1_batch = sn1_an1_batch.to(self.critic_network_device)
                q_sn1_an1_batch = self.critic_network(sn1_an1_batch)

                trajectory_reward_batch = torch.tensor(np.array(list(self.trajectory_reward_memory)), dtype=torch.float32,
                                                       device=self.critic_network_device).view(-1, 1)
                TD_target_batch = trajectory_reward_batch.detach() + pow(self.gamma,self.immediate_reward_memory.maxlen) * q_sn1_an1_batch.detach()

                a1_batch = torch.tensor(np.array(list(self.action_memory)), dtype=torch.float32)
                s1_a1_batch = torch.cat((s1_batch, a1_batch), dim=1)

                s1_a1_batch = s1_a1_batch.to(self.critic_network_device)
                estimate_value_batch = self.critic_network(s1_a1_batch)

                critic_loss = 0.5 * (TD_target_batch - estimate_value_batch)**2
                critic_loss = critic_loss.mean()
                self.critic_loss_queue.append(critic_loss.item())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                s1_batch = s1_batch.to(self.actor_network_device)
                us1_batch = self.actor_network(s1_batch)
                s1_us1_batch = torch.cat((s1_batch, us1_batch), dim=1)

                s1_us1_batch = s1_us1_batch.to(self.critic_network_device)
                q_value = - self.critic_network(s1_us1_batch)
                new_q_sn1_an1_batch = self.critic_network(sn1_an1_batch)
                new_TD_target_batch = (trajectory_reward_batch.detach() +
                                       pow(self.gamma,self.immediate_reward_memory.maxlen) * new_q_sn1_an1_batch)

                actor_loss = - self.TD_error_weight * new_TD_target_batch - (1 - self.TD_error_weight) * q_value
                actor_loss = actor_loss.mean()
                self.actor_loss_queue.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.trajectory_reward_memory.clear()

                if len(self.actor_loss_queue) == self.actor_loss_queue.maxlen:

                    critic_average_loss = sum(self.critic_loss_queue) / self.critic_loss_queue.maxlen
                    self.critic_loss_info = critic_average_loss
                    self.critic_loss_queue.clear()
                    self.critic_scheduler.step(critic_average_loss)

                    actor_average_loss = sum(self.actor_loss_queue) / self.actor_loss_queue.maxlen
                    self.actor_loss_info = actor_average_loss
                    self.actor_loss_queue.clear()
                    self.actor_scheduler.step(actor_average_loss)

        def get_average_loss(self):
            return [self.actor_loss_info,self.critic_loss_info]