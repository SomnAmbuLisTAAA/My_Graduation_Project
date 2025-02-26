from torchviz import make_dot
import math
import torch
import torch.nn as nn
import torch.optim as optim
from unicodedata import normalize

import DeepLearning.DataPreprocessing as DataPreprocessing
from DeepLearning.NN_model import SequentialMultiLayerNN
from DeepLearning.DataPreprocessing import PrioritizedReplayBuffer
import time

state_dim = 4
action_set_dim = 4
observation_dim = 2
action_dim = 2
mini_batch_size = 1024  # Mini-Batch 大小
learning_rate = 0.001   # 学习率
weight_decay = 0.01     # L2 正则化强度（权重衰减因子）
patience = 3            # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5            # 学习率衰减参数

alpha_min = 0.6
beta_min = 0.4
alpha_max = 1.0
beta_max = 1.0
decay_factor = 578

gamma_power_n = 0.95**10

epochs = 100         # 训练轮数

central_critic_network = SequentialMultiLayerNN(state_dim+action_set_dim,128,3,1)
actor_network = SequentialMultiLayerNN(observation_dim,128,3,2)

buffer = PrioritizedReplayBuffer(2**20,state_dim,action_dim,10,0.6,0.4,0.95)
path = 'ReinforcementLearning/Experience/DataSet_12_07_No1_tensors.pth'
buffer.load_from_file(path)

with torch.no_grad():
    loaded_tensors = torch.load(path, weights_only=True)
    trajectory_reward_tensor = loaded_tensors['trajectory_reward_tensor']
    sn1_tensor = loaded_tensors['sn1_tensor']
    s1_a1_tensor = loaded_tensors['s1_a1_tensor']

    trajectory_reward_tensor = trajectory_reward_tensor.float().to('cpu')
    sn1_tensor = sn1_tensor.float().to('cpu')
    s1_a1_tensor = s1_a1_tensor.float().to('cpu')
    o1n1_tensor, o2n1_tensor = torch.split(sn1_tensor, 2, dim=1)

    # print(trajectory_reward_tensor)

# trajectory_reward_tensor_mean, trajectory_reward_tensor_std = DataPreprocessing.get_mean_and_std_of_training_set(trajectory_reward_tensor)
sn1_tensor_mean, sn1_tensor_std = DataPreprocessing.get_mean_and_std_of_training_set(sn1_tensor)
s1_a1_tensor_mean, s1_a1_tensor_std = DataPreprocessing.get_mean_and_std_of_training_set(s1_a1_tensor)
normalized_s1_a1_tensor = (s1_a1_tensor - s1_a1_tensor_mean.to('cpu')) / s1_a1_tensor_std.to('cpu')
normalized_sn1_tensor = (sn1_tensor - sn1_tensor_mean.to('cpu')) / sn1_tensor_std.to('cpu')
normalized_o1n1_tensor, normalized_o2n1_tensor = torch.split(normalized_sn1_tensor, 2, dim=1)

# print("trajectory_reward_tensor_mean, trajectory_reward_tensor_std:\n",trajectory_reward_tensor_mean, trajectory_reward_tensor_std)
print("sn1_tensor_mean, sn1_tensor_std:\n",sn1_tensor_mean, sn1_tensor_std)
print("s1_a1_tensor_mean, s1_a1_tensor_std:\n",s1_a1_tensor_mean, s1_a1_tensor_std)

# 检查是否有 GPU 可用，如果有则使用 GPU 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

central_critic_network = central_critic_network.to(device)
actor_network = actor_network.to(device)

central_critic_network_optimizer = optim.Adam(central_critic_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
actor_network_optimizer = optim.Adam(actor_network.parameters(), lr=learning_rate, weight_decay=weight_decay)

central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer,
                                                                              mode='min', patience=patience , factor=factor)
actor_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer,
                                                                     mode='min', patience=patience , factor=factor)

def update_priorities(uo1n1_tensor_mean, uo1n1_tensor_std):

    with torch.no_grad():
        mean = uo1n1_tensor_mean.to('cpu')
        std = uo1n1_tensor_std.to('cpu')

        actor_network.to('cpu')
        central_critic_network.to('cpu')

        uo1n1_tensor = actor_network(normalized_o1n1_tensor)
        uo2n1_tensor = actor_network(normalized_o2n1_tensor)

        normalized_uo1n1_tensor = (uo1n1_tensor - mean) / std
        normalized_uo2n1_tensor = (uo2n1_tensor - mean) / std

        sn1_actor_an1_tensor = torch.cat((normalized_sn1_tensor, normalized_uo1n1_tensor, normalized_uo2n1_tensor), dim=1)

        q_sn1_actor_an1_tensor = central_critic_network(sn1_actor_an1_tensor)
        q_s1_a1_tensor = central_critic_network(normalized_s1_a1_tensor)
        buffer.update_priorities(q_sn1_actor_an1_tensor.to('cuda'),q_s1_a1_tensor.to('cuda'))

        actor_network.to('cuda')
        central_critic_network.to('cuda')

def critic_network_loss(trajectory_reward_batch, sn1_batch, s1_a1_batch, uo1n1_tensor_mean, uo1n1_tensor_std):

    with torch.no_grad():

        o1n1_batch, o2n1_batch = torch.split(sn1_batch, 2, dim=1)

        uo1n1_batch = actor_network(o1n1_batch)
        uo2n1_batch = actor_network(o2n1_batch)

        normalized_uo1n1_batch = (uo1n1_batch - uo1n1_tensor_mean) / uo1n1_tensor_std
        normalized_uo2n1_batch = (uo2n1_batch - uo1n1_tensor_mean) / uo1n1_tensor_std

        sn1_actor_an1_batch = torch.cat((sn1_batch, normalized_uo1n1_batch, normalized_uo2n1_batch), dim=1)
        q_sn1_actor_an1_batch = central_critic_network(sn1_actor_an1_batch)
        # print("q_sn1_actor_an1_batch:\n",q_sn1_actor_an1_batch)

    q_s1_a1_batch = central_critic_network(s1_a1_batch)
    # print("q_s1_a1_batch:\n",q_s1_a1_batch)
    TD_error = trajectory_reward_batch + gamma_power_n * q_sn1_actor_an1_batch - q_s1_a1_batch
    # print("TD_error:\n",TD_error)
    critic_loss_tensor = 0.5 * TD_error**2
    # print(critic_loss)
    # make_dot(critic_loss, params={"s1_a1_batch": s1_a1_batch}, show_attrs=True).render("critic_loss_compute_graph", format="svg")
    return critic_loss_tensor

def actor_network_loss(trajectory_reward_batch, sn1_batch, s1_a1_batch, uo1n1_tensor_mean, uo1n1_tensor_std):

    with torch.no_grad():
        o1n1_batch, o2n1_batch = torch.split(sn1_batch, 2, dim=1)
        uo1n1_batch = actor_network(o1n1_batch)
        uo2n1_batch = actor_network(o2n1_batch)

        normalized_uo1n1_batch = (uo1n1_batch - uo1n1_tensor_mean) / uo1n1_tensor_std
        normalized_uo2n1_batch = (uo2n1_batch - uo1n1_tensor_mean) / uo1n1_tensor_std

        sn1_actor_an1_batch = torch.cat((sn1_batch, normalized_uo1n1_batch, normalized_uo2n1_batch), dim=1)
        q_sn1_actor_an1_batch = central_critic_network(sn1_actor_an1_batch)

        s1_batch, a1_batch = torch.split(s1_a1_batch, 4, dim=1)
        o1_batch, o2_batch = torch.split(s1_batch, 2, dim=1)
        # print(o1_batch.requires_grad)
        # print(o2_batch.requires_grad)

    uo1_batch = actor_network(o1_batch)
    uo2_batch = actor_network(o2_batch)

    normalized_uo1_batch = (uo1_batch - uo1n1_tensor_mean) / uo1n1_tensor_std
    normalized_uo2_batch = (uo2_batch - uo1n1_tensor_mean) / uo1n1_tensor_std

    # print(uo1_batch.requires_grad)
    # print(uo2_batch.requires_grad)
    s1_us1_batch = torch.cat((s1_batch, normalized_uo1_batch, normalized_uo2_batch), dim=1)
    # print(s1_us1_batch.requires_grad)
    q_s1_us1_batch = central_critic_network(s1_us1_batch)
    # print(q_s1_us1_batch.requires_grad)

    advantage_function = trajectory_reward_batch + gamma_power_n * q_sn1_actor_an1_batch - q_s1_us1_batch
    actor_loss_tensor = -advantage_function
    # print('q_s1_us1_batch:\n',q_s1_us1_batch)

    return actor_loss_tensor

def training():

    with torch.no_grad():
        actor_network.to('cpu')
        uo1n1_tensor = actor_network(normalized_o1n1_tensor)
        uo1n1_tensor_mean, uo1n1_tensor_std = DataPreprocessing.get_mean_and_std_of_training_set(uo1n1_tensor)
        actor_network.to('cuda')

    update_priorities(uo1n1_tensor_mean, uo1n1_tensor_std)

    central_critic_network.train()
    actor_network.train()
    critic_running_loss = 0.0
    actor_running_loss = 0.0
    start_time = time.time()

    for epoch in range(epochs):

        with torch.no_grad():

            actor_network.to('cpu')
            uo1n1_tensor = actor_network(normalized_o1n1_tensor)
            uo1n1_tensor_mean, uo1n1_tensor_std = DataPreprocessing.get_mean_and_std_of_training_set(uo1n1_tensor)
            actor_network.to('cuda')

            s1_a1_batch, sn1_batch, trajectory_reward_batch, weight_batch = buffer.sample(mini_batch_size)
            s1_a1_batch = s1_a1_batch.float()
            sn1_batch = sn1_batch.float()
            trajectory_reward_batch = trajectory_reward_batch.float()
            weight_batch = weight_batch.float()

            normalized_s1_a1_batch = (s1_a1_batch - s1_a1_tensor_mean) / s1_a1_tensor_std
            normalized_sn1_batch = (sn1_batch - sn1_tensor_mean) / sn1_tensor_std

        # print(sn1_batch)
        # print('trajectory_reward_batch:\n',trajectory_reward_batch)
        # print(weight_batch)

        critic_loss = critic_network_loss(trajectory_reward_batch, normalized_sn1_batch, normalized_s1_a1_batch, uo1n1_tensor_mean, uo1n1_tensor_std)
        weighted_critic_loss = (weight_batch * critic_loss).mean()

        central_critic_network_optimizer.zero_grad()
        weighted_critic_loss.backward()
        central_critic_network_optimizer.step()
        critic_running_loss += weighted_critic_loss.item()

        # print(normalized_s1_a1_batch.requires_grad)
        actor_loss = actor_network_loss(trajectory_reward_batch, normalized_sn1_batch, normalized_s1_a1_batch, uo1n1_tensor_mean, uo1n1_tensor_std)
        weighted_actor_loss = (weight_batch * actor_loss).mean()

        actor_network_optimizer.zero_grad()
        weighted_actor_loss.backward()
        actor_network_optimizer.step()
        actor_running_loss += weighted_actor_loss.item()

        if (epoch + 1) % 100 == 0:
            increase = 1 - math.exp(-(epoch+1) / decay_factor)
            alpha = alpha_min + (alpha_max - alpha_min) * increase
            beta = beta_min + (beta_max - beta_min) * increase
            print('alpha,beta:',alpha,beta)
            buffer.change_alpha(alpha)
            buffer.change_beta(beta)

        if (epoch+1)%10 == 0:

            avg_critic_loss = critic_running_loss / 10
            central_critic_network_scheduler.step(avg_critic_loss)
            critic_running_loss = 0.0
            avg_actor_loss = actor_running_loss / 10
            actor_network_scheduler.step(avg_actor_loss)
            actor_running_loss = 0.0

            print('avg_critic_loss,avg_actor_loss:',avg_critic_loss,avg_actor_loss)
            update_priorities(uo1n1_tensor_mean, uo1n1_tensor_std)

            end_time = time.time()
            print(f"Rounds: [{100 * (epoch + 1) / epochs:.2f}%]")
            elapsed_time = end_time - start_time
            print(f"10步运行时间: {elapsed_time:.6f} 秒")
            start_time = time.time()





training()

path = ["DeepLearning/model/12_08/actor_network_02.pth",
             "DeepLearning/model/12_08/central_critic_network_02.pth"]
torch.save(actor_network.state_dict(), path[0])
torch.save(central_critic_network.state_dict(), path[1])

actor_network.eval()
actor_network.to('cpu')
uo_tensor = actor_network(normalized_o1n1_tensor)
mean, std = DataPreprocessing.get_mean_and_std_of_training_set(uo_tensor)
print('mean, std:',mean, std)