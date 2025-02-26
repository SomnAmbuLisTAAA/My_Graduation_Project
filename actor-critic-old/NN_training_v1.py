import time

import numpy as np
import torch
import torch.optim as optim

from DeepLearning.NN_model import SequentialMultiLayerNN, SequentialMultiLayerNN_with_scaled_tanh
from DeepLearning.DataPreprocessing import PrioritizedReplayBuffer

from collections import deque

state_dim = 4
action_set_dim = 4
observation_dim = 2
action_dim = 2

learning_rate = 0.1     # 学习率
weight_decay = 0.01     # L2 正则化强度（权重衰减因子）
patience = 3            # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5            # 学习率衰减参数

gamma_power_n = 0.95**10

central_critic_network = SequentialMultiLayerNN(state_dim+action_set_dim,128,3,1)
actor_network = SequentialMultiLayerNN_with_scaled_tanh(observation_dim,128,3,2,10,0.1)

central_critic_network_cpu = SequentialMultiLayerNN(state_dim+action_set_dim,128,3,1)
actor_network_cpu = SequentialMultiLayerNN_with_scaled_tanh(observation_dim,128,3,2,10,0.1)
central_critic_network_cpu.to('cpu')
actor_network_cpu.to('cpu')

buffer = PrioritizedReplayBuffer(2**20,state_dim,action_dim,10,0.6,0.4,0.95)
path = 'ReinforcementLearning/Experience/DataSet_12_07_No1_tensors.pth'
buffer.load_from_file(path)

with torch.no_grad():
    loaded_tensors = torch.load(path, weights_only=True)
    trajectory_reward_tensor_cpu = loaded_tensors['trajectory_reward_tensor']
    sn1_tensor_cpu = loaded_tensors['sn1_tensor']
    s1_a1_tensor_cpu = loaded_tensors['s1_a1_tensor']

    trajectory_reward_tensor_cpu = trajectory_reward_tensor_cpu.float().to('cpu')
    sn1_tensor_cpu = sn1_tensor_cpu.float().to('cpu')
    s1_a1_tensor_cpu = s1_a1_tensor_cpu.float().to('cpu')

    '''trajectory_reward_tensor_cpu = trajectory_reward_tensor_cpu.float()
    sn1_tensor_cpu = sn1_tensor_cpu.float()
    s1_a1_tensor_cpu = s1_a1_tensor_cpu.float()'''

    o1n1_tensor_cpu, o2n1_tensor_cpu = torch.split(sn1_tensor_cpu, 2, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
central_critic_network = central_critic_network.to(device)
actor_network = actor_network.to(device)

central_critic_network_optimizer = optim.Adam(central_critic_network.parameters(), lr=0.01, weight_decay=weight_decay)
actor_network_optimizer = optim.Adam(actor_network.parameters(), lr=0.1, weight_decay=weight_decay)


def update_priorities_of_buffer():

    with torch.no_grad():

        central_critic_network_cpu.load_state_dict(central_critic_network.state_dict())
        actor_network_cpu.load_state_dict(actor_network.state_dict())

        q_s1_a1_tensor_cpu = central_critic_network_cpu(s1_a1_tensor_cpu)

        uo1n1_tensor_cpu = actor_network_cpu(o1n1_tensor_cpu)
        uo2n1_tensor_cpu = actor_network_cpu(o2n1_tensor_cpu)

        sn1_an1_tensor_cpu = torch.cat((sn1_tensor_cpu, uo1n1_tensor_cpu, uo2n1_tensor_cpu), dim=1)
        q_sn1_an1_tensor_cpu = central_critic_network_cpu(sn1_an1_tensor_cpu)

        buffer.update_priorities(q_sn1_an1_tensor_cpu.to('cuda'), q_s1_a1_tensor_cpu.to('cuda'))


def get_critic_loss(s1_a1_batch, TD_target_batch, weights_batch):

    q_s1_a1_batch = central_critic_network(s1_a1_batch)
    critic_loss = 0.5*(TD_target_batch - q_s1_a1_batch)**2
    return (critic_loss * weights_batch).mean()


def get_actor_loss_minus_q(s1_batch,weights_batch):

    o11_batch, o21_batch = torch.split(s1_batch, 2, dim=1)

    uo11_batch = actor_network(o11_batch)
    uo21_batch = actor_network(o21_batch)

    s1_us1_batch = torch.cat((s1_batch, uo11_batch, uo21_batch), dim=1)
    q_s1_us1_batch = central_critic_network(s1_us1_batch)
    actor_loss = - q_s1_us1_batch
    return (actor_loss * weights_batch).mean()


def get_actor_loss_advantage_function(sn1_batch, s1_batch, trajectory_reward_batch, weights_batch):

    o1n1_batch, o2n1_batch = torch.split(sn1_batch, 2, dim=1)
    o11_batch, o21_batch = torch.split(s1_batch, 2, dim=1)

    uo1n1_batch = actor_network(o1n1_batch)
    uo2n1_batch = actor_network(o2n1_batch)

    uo11_batch = actor_network(o11_batch)
    uo21_batch = actor_network(o21_batch)

    sn1_usn1_batch = torch.cat((sn1_batch, uo1n1_batch, uo2n1_batch), dim=1)
    s1_us1_batch = torch.cat((s1_batch, uo11_batch, uo21_batch), dim=1)

    q_sn1_usn1_batch = central_critic_network(sn1_usn1_batch)
    q_s1_us1_batch = central_critic_network(s1_us1_batch)

    advantage_function = trajectory_reward_batch + gamma_power_n * q_sn1_usn1_batch - q_s1_us1_batch
    actor_loss = -advantage_function
    return (actor_loss * weights_batch).mean()


def critic_iteration(s1_a1_batch, TD_target_batch, weights_batch):

    max_update_times = 100
    relative_threshold = 1e-4
    prev_best_loss = float('inf')
    central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer,
                                                                                  mode='min', patience=patience,
                                                                                  threshold=1e-3, factor=factor)

    for update_time in range(max_update_times):
        critic_loss = get_critic_loss(s1_a1_batch, TD_target_batch, weights_batch)
        central_critic_network_optimizer.zero_grad()
        critic_loss.backward()
        central_critic_network_optimizer.step()

        central_critic_network_scheduler.step(critic_loss.item())

        # 计算相对损失变化（基于历史最佳损失）
        if prev_best_loss != float('inf'):
            relative_change = (prev_best_loss - critic_loss.item()) / prev_best_loss
        else:
            relative_change = float('inf')  # 第一次循环无相对变化

        # 更新历史最佳损失
        if critic_loss.item() < prev_best_loss:
            prev_best_loss = critic_loss.item()

        # 判断是否收敛
        if 0 < relative_change < relative_threshold:
            # print("critic_iteration_times:",update_time+1)
            return critic_loss.item()

    # print("critic_iteration_times:", update_time+1)
    return critic_loss.item()


def actor_iteration(sn1_batch, s1_batch, trajectory_reward_batch, weights_batch):
    max_update_times = 100
    relative_threshold = 1e-6
    prev_best_loss = float('inf')

    actor_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer,
                                                                        mode='min', patience=patience,
                                                                        threshold=1e-5, factor=factor)

    for update_time in range(max_update_times):
        actor_loss = get_actor_loss_minus_q(s1_batch, weights_batch)
        # actor_loss = get_actor_loss_advantage_function(sn1_batch, s1_batch, trajectory_reward_batch, weights_batch)
        actor_network_optimizer.zero_grad()
        actor_loss.backward()
        actor_network_optimizer.step()

        '''for param_group in actor_network_optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")'''

        actor_network_scheduler.step(actor_loss.item())

        # 计算相对损失变化（基于历史最佳损失）
        if prev_best_loss != float('inf'):
            relative_change = (prev_best_loss - actor_loss.item()) / prev_best_loss
        else:
            relative_change = float('inf')  # 第一次循环无相对变化

        # 更新历史最佳损失
        if actor_loss.item() < prev_best_loss:
            prev_best_loss = actor_loss.item()

        # print("actor_loss",actor_loss.item())

        # 判断是否收敛
        if 0 < relative_change < relative_threshold:
            # print("actor_iteration_times:",update_time+1,"\n")
            return actor_loss.item()

    # print("actor_iteration_times:", update_time+1, "\n")
    return actor_loss.item()


def step(batch_size):

    with torch.no_grad():

        s1_a1_batch, sn1_batch, trajectory_reward_batch, TD_target_batch, weights_batch = buffer.sample(batch_size)

        s1_a1_batch = s1_a1_batch.float()
        sn1_batch = sn1_batch.float()
        trajectory_reward_batch = trajectory_reward_batch.float()
        weights_batch = weights_batch.float()
        TD_target_batch = TD_target_batch.float()

        s1_batch, a1_batch = torch.split(s1_a1_batch, 4, dim=1)

    critic_loss = critic_iteration(s1_a1_batch, TD_target_batch, weights_batch)
    actor_loss = actor_iteration(sn1_batch, s1_batch, trajectory_reward_batch, weights_batch)

    return actor_loss, critic_loss


def update_alpha_beta_of_buffer(new_alpha, new_beta):
    buffer.change_alpha(new_alpha)
    buffer.change_beta(new_beta)

def calculate_alpha_beta(step_count, total_steps, alpha_max, alpha_min, beta_max, beta_min, decay_factor):
    # 计算百分比（当前轮数占总训练轮数的比例）
    percentage = step_count / total_steps

    # 根据百分比计算alpha和beta
    alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-decay_factor * percentage)
    beta = beta_min + (beta_max - beta_min) * (1 - np.exp(-decay_factor * percentage))

    return alpha, beta

def adjust_alpha_beta_sigmoid(t, T, alpha_max, alpha_min, beta_max, beta_min, k=20, p_m=0.5):
    p = t / T  # Normalize step
    alpha = alpha_max - (alpha_max - alpha_min) / (1 + np.exp(-k * (p - p_m)))
    beta = beta_min + (beta_max - beta_min) / (1 + np.exp(-k * (p - p_m)))
    return alpha,beta

def main():
    rounds = 10000
    alpha_min = 0.0
    alpha_max = 0.6
    beta_min = 0.4
    beta_max = 1.0
    decay_factor = 1.25
    mini_batch_size = 1024  # Mini-Batch 大小

    max_length = 10
    actor_loss_deque = deque(maxlen=max_length)
    critic_loss_deque = deque(maxlen=max_length)

    update_priorities_of_buffer()

    central_critic_network.train()
    actor_network.train()

    start_time = time.time()

    for round in range(rounds):

        # print(mini_batch_size)
        actor_loss, critic_loss = step(mini_batch_size)

        actor_loss_deque.append(actor_loss)
        critic_loss_deque.append(critic_loss)

        '''if 0.1 <= (round+1)/rounds < 0.4:
            mini_batch_size = 128

        if (round+1)/rounds >= 0.4:
            mini_batch_size = 1024'''

        if len(actor_loss_deque) == actor_loss_deque.maxlen:

            # start_time = time.time()
            update_priorities_of_buffer()
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"优先级更新用时: {elapsed_time:.6f} 秒\n")

            avg_actor_loss = sum(actor_loss_deque) / max_length
            avg_critic_loss = sum(critic_loss_deque) / max_length

            actor_loss_deque.clear()
            critic_loss_deque.clear()

            # alpha, beta = calculate_alpha_beta(round + 1, rounds, alpha_max, alpha_min, beta_max, beta_min, decay_factor)
            alpha, beta = adjust_alpha_beta_sigmoid(round + 1, rounds, alpha_max, alpha_min, beta_max, beta_min)
            update_alpha_beta_of_buffer(alpha, beta)

            print("avg_actor_loss, avg_critic_loss:", avg_actor_loss, avg_critic_loss)
            print("alpha, beta", alpha, beta)

            end_time = time.time()
            print(f"Rounds: [{100 * (round + 1) / rounds:.2f}%]")
            elapsed_time = end_time - start_time
            print(f"10步运行时间: {elapsed_time:.6f} 秒\n")
            start_time = time.time()

    path = ["DeepLearning/model/12_13/actor_network_04.pth",
            "DeepLearning/model/12_13/central_critic_network_04.pth"]
    torch.save(actor_network.state_dict(), path[0])
    torch.save(central_critic_network.state_dict(), path[1])




main()