from math import gamma
import numpy as np
from collections import deque
from DeepLearning.NN_model import SequentialMultiLayerNN
import random
import ReinforcementLearning.Agent as agent
import torch
import torch.nn as nn
import torch.optim as optim

class DQN:
    class Agent:
        def __init__(self, state_dim, action_dim, agent_radius, vx_min, vx_max, vy_min, vy_max, init_pos_x, init_pos_y, gamma=0.99):

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma  # 折扣因子
            self.memory = deque(maxlen=1000000)  # 经验回放缓冲区
            self.vx_max = abs(vx_max)
            self.vy_max = abs(vy_max)
            self.vx_min = vx_min
            self.vy_min = vy_min
            self.speed_limits = (self.vx_min, self.vx_max, self.vy_min, self.vy_max)
            self.agent_radius = agent_radius

            self.init_pos = np.array([init_pos_x,init_pos_y])
            self.pos = np.array([init_pos_x,init_pos_y])
            self.v = np.array([0,0])
            self.LastPos = np.array([init_pos_x,init_pos_y])
            self.reward = 0.0
            self.step = -1

            self.policy_network = SequentialMultiLayerNN(state_dim,2,128, action_dim)# 创建Q网络
            self.target_policy_network = self.policy_network

        def agent_init(self):
            self.v = np.array([0, 0])
            self.LastPos = self.init_pos
            self.pos = self.init_pos
            self.reward = 0.0
            self.step = -1

        def agent_default_init(self):
            self.v = np.array([0, 0])
            self.LastPos = np.array([self.agent_radius, self.agent_radius])
            self.LastPos = np.array([self.agent_radius, self.agent_radius])
            self.step = -1
            self.reward = 0.0

        def get_agent_pos(self):
            return self.pos

        def explore_act(self):
            self.v = np.array([random.uniform(self.vx_min, self.vx_max), random.uniform(self.vy_min, self.vy_max)])

        def policy_act(self):
            input = torch.from_numpy(self.pos).float()
            input_data = input.unsqueeze(0)
            #print(f"input_data:{input_data}")
            self.policy_network.eval()
            output_data = self.policy_network(input_data)
            output = output_data.detach().cpu().numpy().flatten()
            #print(f"output:{output}")
            self.v = np.clip(output, [self.vx_min, self.vy_min], [self.vx_max, self.vy_max])
            #print(f"v:{self.v}\n")

        def upgrade_pos(self,sample_time):
            self.step += 1
            self.LastPos = self.pos
            self.pos = self.pos + sample_time * self.v

        def get_agent_radius(self):
            return self.agent_radius

        def undo_pos(self):
            self.pos = self.LastPos

        def write_reward(self,reward):
            self.reward += reward * pow(self.gamma, self.step)

        def get_reward(self):
            return self.reward

        def get_last_pos(self):
            return self.LastPos

        def get_speed_limits(self):
            return self.speed_limits

class Actor_Critic:
    class simplest_ac_agent:
        def __init__(self, state_dim, action_dim, agent_radius, vx_min, vx_max, vy_min, vy_max, init_pos_x, init_pos_y, gamma=0.99):

            self.state_dim = state_dim
            self.action_dim = action_dim

            # 折扣因子
            self.gamma = gamma

            # 经验回放缓冲区
            self.action_memory = deque(maxlen=10)
            self.immediate_reward_memory = deque(maxlen=10)
            self.immediate_reward = 0

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

            self.actor_network = SequentialMultiLayerNN(state_dim,2,128, action_dim)
            self.critic_network = SequentialMultiLayerNN(state_dim + action_dim,2,128, 1)
            self.critic_loss = 0.0
            self.actor_loss = 0.0
            self.actor_loss_queue = deque(maxlen=10)
            self.critic_loss_queue = deque(maxlen=10)
            self.TD_error = 0.0

            self.actor_average_loss = deque(maxlen=10)
            self.critic_average_loss = deque(maxlen=10)

            self.actor_loss_info = 0.0
            self.critic_loss_info = 0.0

            self.learning_rate = 0.001  # 学习率
            self.batch_size = 64  # Mini-Batch 大小
            self.epochs = 100  # 训练轮数
            self.weight_decay = 0.01  # L2 正则化强度（权重衰减因子）
            self.patience = 3  # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
            self.factor = 0.5  # 学习率衰减参数

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
            self.step = -1
            self.action_memory.clear()
            self.immediate_reward_memory.clear()

        def agent_random_init(self,width,height):
            self.v = np.array([0, 0])
            self.init_pos = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
            self.LastPos = self.init_pos
            self.pos = self.init_pos
            self.reward = 0.0
            self.step = -1
            self.action_memory.clear()
            self.immediate_reward_memory.clear()

        def get_agent_pos(self):
            return self.pos

        def explore_act(self):
            self.v = np.array([random.uniform(self.vx_min, self.vx_max), random.uniform(self.vy_min, self.vy_max)])

        def policy_act(self,state):

            input_data = state
            input_data = input_data.to(self.actor_network_device)
            # print(f"input_data:{input_data}")
            self.actor_network.eval()
            with torch.no_grad():
                output_data = self.actor_network(input_data)
                output = output_data.detach().cpu().numpy().flatten()
                # print(f"output:{output}")
                self.v = np.clip(output, [self.vx_min, self.vy_min], [self.vx_max, self.vy_max])
                # print(f"v:{self.v}\n")

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

        def save_experience(self,reward):
            self.immediate_reward = reward
            self.reward += self.immediate_reward * pow(self.gamma,self.step)
            self.action_memory.append(self.v)
            self.immediate_reward_memory.append(self.immediate_reward)
            return len(self.action_memory) ==self.action_memory.maxlen

        def calculate_critic_loss(self,s1,sn1):
            if len(self.action_memory) == self.action_memory.maxlen:
                self.actor_network.eval()
                self.critic_network.eval()

                with torch.no_grad():
                    trajectory_reward = 0
                    for i in range(len(self.immediate_reward_memory)):
                        trajectory_reward += self.immediate_reward_memory[i] * pow(self.gamma, i)

                    a1 = self.action_memory[0]
                    a1 = torch.tensor(a1, dtype=torch.float32).view(1, -1)

                    actor_input_data = sn1
                    actor_input_data = actor_input_data.to(self.actor_network_device)
                    actor_output_data = self.actor_network(actor_input_data)
                    # print(actor_output_data)

                    actor_output = actor_output_data.detach().cpu().numpy().flatten()
                    actor_output = np.clip(actor_output, [self.vx_min, self.vy_min], [self.vx_max, self.vy_max])

                    actor_output = torch.tensor(actor_output, dtype=torch.float32).view(1, -1)
                    critic_input_data = torch.cat((sn1, actor_output), dim=1)

                    critic_input_data = critic_input_data.to(self.critic_network_device)
                    critic_output_data = self.critic_network(critic_input_data)

                    critic_output = critic_output_data.detach().cpu().numpy().flatten()

                    TD_target = trajectory_reward + critic_output * pow(self.gamma, len(self.immediate_reward_memory))

                    critic_input_data = torch.cat((s1, a1), dim=1)

                    critic_input_data = critic_input_data.to(self.critic_network_device)
                    critic_output_data = self.critic_network(critic_input_data)

                    critic_output = critic_output_data.detach().cpu().numpy().flatten()

                    self.TD_error = TD_target - critic_output
                    self.critic_loss = 0.5 * pow(self.TD_error, 2)
                    self.critic_loss_queue.append(self.critic_loss)


        def calculate_actor_loss(self,s1):
            if len(self.action_memory) == self.action_memory.maxlen:
                self.actor_network.eval()
                self.critic_network.eval()

                with torch.no_grad():

                    actor_input_data = s1

                    actor_input_data = actor_input_data.to(self.actor_network_device)
                    actor_output_data = self.actor_network(actor_input_data)

                    actor_output = actor_output_data.detach().cpu().numpy().flatten()
                    actor_output = np.clip(actor_output, [self.vx_min, self.vy_min], [self.vx_max, self.vy_max])

                    actor_output = torch.tensor(actor_output, dtype=torch.float32).view(1, -1)
                    critic_input_data = torch.cat((s1, actor_output), dim=1)

                    critic_input_data = critic_input_data.to(self.critic_network_device)
                    critic_output_data = self.critic_network(critic_input_data)
                    critic_output = critic_output_data.detach().cpu().numpy().flatten()

                    self.actor_loss = - critic_output
                    # print(self.actor_loss)
                    self.actor_loss_queue.append(self.actor_loss)

        def update_critic_network(self):
            if len(self.critic_loss_queue) == self.critic_loss_queue.maxlen:
                self.critic_network.train()

                critic_loss_array = np.array(self.critic_loss_queue)
                critic_loss_tensor = torch.tensor(critic_loss_array, dtype=torch.float32, requires_grad=True)
                critic_loss = torch.mean(critic_loss_tensor)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.critic_average_loss.append(critic_loss.item())
                # print(self.critic_average_loss)

            if len(self.critic_average_loss) == self.critic_average_loss.maxlen:
                # print("critic_average_loss",self.critic_average_loss)
                critic_average_loss = sum(self.critic_average_loss) / self.actor_average_loss.maxlen
                self.critic_loss_info = critic_average_loss
                self.critic_average_loss.clear()
                self.critic_scheduler.step(critic_average_loss)

        def update_actor_network(self):
            if len(self.actor_loss_queue) == self.actor_loss_queue.maxlen:
                self.actor_network.train()

                actor_loss_array = np.array(self.actor_loss_queue)
                actor_loss_tensor = torch.tensor(actor_loss_array, dtype=torch.float32, requires_grad=True)
                actor_loss = torch.mean(actor_loss_tensor)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.actor_average_loss.append(actor_loss.item())

            if len(self.actor_average_loss) == self.actor_average_loss.maxlen:
                print("actor_average_loss",self.actor_average_loss)
                actor_average_loss = sum(self.actor_average_loss) / self.actor_average_loss.maxlen
                self.actor_loss_info = actor_average_loss
                self.actor_average_loss.clear()
                self.actor_scheduler.step(actor_average_loss)

        def get_average_loss(self):
            return [self.actor_loss_info, self.critic_loss_info]

        def upgrade_network(self,s1_batch,sn1_batch):

            self.actor_network.train()
            self.critic_network.train()