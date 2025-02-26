import time
import numpy as np
import torch
import torch.optim as optim
from DeepLearning.NN_model import SequentialMultiLayerNN, SequentialMultiLayerNN_with_scaled_tanh
from DeepLearning.DataPreprocessing import PrioritizedReplayBuffer
from collections import deque

buffer_device = 'cuda'

agent_num = 2
task_num = 2
agent_pos_dim = 2
agent_action_dim = 2

state_dim = agent_pos_dim * agent_num + task_num
action_dim = agent_action_dim * agent_num
observation_dim = agent_pos_dim + task_num

weight_decay = 0.01     # L2 正则化强度（权重衰减因子）
patience = 3            # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5            # 学习率衰减参数
gamma_power_n = 0.99**10

max_iteration_times = 100
learning_rate = 0.01

central_critic_network = SequentialMultiLayerNN(state_dim+action_dim,128,3,1)
actor_network = SequentialMultiLayerNN_with_scaled_tanh(observation_dim,128,3,agent_action_dim,1,0.1)
central_critic_network = central_critic_network.to('cuda')
actor_network = actor_network.to('cuda')

central_critic_network_cpu = SequentialMultiLayerNN(state_dim+action_dim,128,3,1)
actor_network_cpu = SequentialMultiLayerNN_with_scaled_tanh(observation_dim,128,3,agent_action_dim,1,0.1)
central_critic_network_cpu.to('cpu')
actor_network_cpu.to('cpu')

buffer = PrioritizedReplayBuffer(2**20,agent_pos_dim,agent_action_dim,agent_num,task_num,10,0.6,0.4,0.99,device=buffer_device)
path = 'ReinforcementLearning/Experience/DataSet_12_25_No3_tensors.pth'
buffer.load_from_file(path)

with torch.no_grad():

    loaded_tensors = torch.load(path, weights_only=True, map_location='cpu')
    for key, tensor in loaded_tensors.items():
        loaded_tensors[key] = tensor.float()

    trajectory_reward_tensor_cpu = loaded_tensors['trajectory_reward_tensor']
    agent_pos_n1_tensor_cpu = loaded_tensors['agent_pos_n1_tensor']
    agent_pos_1_tensor_cpu = loaded_tensors['agent_pos_1_tensor']
    agent_action_1_tensor_cpu = loaded_tensors['agent_action_1_tensor']
    tasks_state_n1_tensor_cpu = loaded_tensors['tasks_state_n1_tensor']
    tasks_state_1_tensor_cpu = loaded_tensors['tasks_state_1_tensor']

    agent1_pos_n1_tensor_cpu, agent2_pos_n1_tensor_cpu = torch.split(agent_pos_n1_tensor_cpu, 2, dim=1)

    o1_n1_tensor_cpu = torch.cat((agent1_pos_n1_tensor_cpu, tasks_state_n1_tensor_cpu), dim=1)
    o2_n1_tensor_cpu = torch.cat((agent2_pos_n1_tensor_cpu, tasks_state_n1_tensor_cpu), dim=1)

    s_1_a_1_tensor_cpu = torch.cat((agent_pos_1_tensor_cpu, tasks_state_1_tensor_cpu, agent_action_1_tensor_cpu), dim=1)
    s_n1_tensor_cpu = torch.cat((agent_pos_n1_tensor_cpu, tasks_state_n1_tensor_cpu), dim=1)

central_critic_network_optimizer = optim.Adam(central_critic_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
actor_network_optimizer = optim.Adam(actor_network.parameters(), lr=learning_rate, weight_decay=weight_decay)


def update_priorities_of_buffer():

    with torch.no_grad():

        central_critic_network_cpu.load_state_dict(central_critic_network.state_dict())
        actor_network_cpu.load_state_dict(actor_network.state_dict())

        q_s_1_a_1_tensor_cpu = central_critic_network_cpu(s_1_a_1_tensor_cpu)

        u_o1_n1_tensor_cpu = actor_network_cpu(o1_n1_tensor_cpu)
        u_o2_n1_tensor_cpu = actor_network_cpu(o2_n1_tensor_cpu)

        s_n1_a_n1_tensor_cpu = torch.cat((s_n1_tensor_cpu, u_o1_n1_tensor_cpu, u_o2_n1_tensor_cpu), dim=1)
        q_s_n1_a_n1_tensor_cpu = central_critic_network_cpu(s_n1_a_n1_tensor_cpu)

        buffer.update_priorities(q_s_n1_a_n1_tensor_cpu.to(buffer_device), q_s_1_a_1_tensor_cpu.to(buffer_device))


def get_critic_loss(s_1_a_1_batch, trajectory_reward_batch, s_n1_batch, o1_n1_batch, o2_n1_batch, weights_batch):

    with torch.no_grad():
        u_o1_n1_batch = actor_network(o1_n1_batch)
        u_o2_n1_batch = actor_network(o2_n1_batch)
        s_n1_u_s_n1_batch = torch.cat((s_n1_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
        q_s_n1_u_s_n1_batch = central_critic_network(s_n1_u_s_n1_batch)
        TD_target_batch = trajectory_reward_batch + gamma_power_n * q_s_n1_u_s_n1_batch

    q_s_1_a_1_batch = central_critic_network(s_1_a_1_batch)
    critic_loss = 0.5*(TD_target_batch - q_s_1_a_1_batch)**2
    return (critic_loss * weights_batch).mean()


def get_actor_loss_minus_q(agent_pos_1_batch, tasks_state_1_batch, o1_1_batch,o2_1_batch, weights_batch):

    u_o1_1_batch = actor_network(o1_1_batch)
    u_o2_1_batch = actor_network(o2_1_batch)

    s_1_u_s_1_batch = torch.cat((agent_pos_1_batch, tasks_state_1_batch, u_o1_1_batch, u_o2_1_batch), dim=1)
    q_s_1_u_s_1_batch = central_critic_network(s_1_u_s_1_batch)
    actor_loss = - q_s_1_u_s_1_batch
    return (actor_loss * weights_batch).mean()


def get_actor_loss_advantage_function(agent_pos_n1_batch, agent_pos_1_batch, tasks_state_n1_batch, tasks_state_1_batch,
                                      o1_n1_batch, o2_n1_batch, o1_1_batch, o2_1_batch,
                                      trajectory_reward_batch, weights_batch):

    u_o1_n1_batch = actor_network(o1_n1_batch)
    u_o2_n1_batch = actor_network(o2_n1_batch)
    u_o1_1_batch = actor_network(o1_1_batch)
    u_o2_1_batch = actor_network(o2_1_batch)

    s_n1_u_s_n1_batch = torch.cat((agent_pos_n1_batch, tasks_state_n1_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
    s_1_u_s_1_batch = torch.cat((agent_pos_1_batch, tasks_state_1_batch, u_o1_1_batch, u_o2_1_batch), dim=1)

    q_s_n1_u_s_n1_batch = central_critic_network(s_n1_u_s_n1_batch)
    q_s_1_u_s_1_batch = central_critic_network(s_1_u_s_1_batch)

    advantage_function = trajectory_reward_batch + gamma_power_n * q_s_n1_u_s_n1_batch - q_s_1_u_s_1_batch
    actor_loss = -advantage_function
    return (actor_loss * weights_batch).mean()


def critic_iteration(agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch, trajectory_reward_batch,
                     agent_pos_n1_batch, tasks_state_n1_batch, weights_batch):

    max_update_times = max_iteration_times
    relative_threshold = 1e-5
    prev_best_loss = float('inf')

    for param_group in central_critic_network_optimizer.param_groups:
        param_group['lr'] = learning_rate
    central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer,
                                                                                  mode='min', patience=patience,
                                                                                  threshold=1e-4, factor=factor)

    with torch.no_grad():
        agent1_pos_n1, agent2_pos_n1 = torch.split(agent_pos_n1_batch, 2, dim=1)
        o1_n1_batch = torch.cat((agent1_pos_n1, tasks_state_n1_batch), dim=1)
        o2_n1_batch = torch.cat((agent2_pos_n1, tasks_state_n1_batch), dim=1)
        s_1_a_1_batch = torch.cat((agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch), dim=1)
        s_n1_batch = torch.cat((agent_pos_n1_batch, tasks_state_n1_batch), dim=1)

    for update_time in range(max_update_times):

        critic_loss = get_critic_loss(s_1_a_1_batch, trajectory_reward_batch, s_n1_batch, o1_n1_batch, o2_n1_batch, weights_batch)

        central_critic_network_optimizer.zero_grad()
        critic_loss.backward()
        central_critic_network_optimizer.step()

        '''for param_group in central_critic_network_optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")'''

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


def actor_iteration(agent_pos_n1_batch, tasks_state_n1_batch, agent_pos_1_batch, tasks_state_1_batch,
                    trajectory_reward_batch, weights_batch, advantage_function = False):

    max_update_times = max_iteration_times
    relative_threshold = 1e-7
    prev_best_loss = float('inf')

    for param_group in actor_network_optimizer.param_groups:
        param_group['lr'] = learning_rate
    actor_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer,
                                                                        mode='min', patience=patience,
                                                                        threshold=1e-6, factor=factor)

    with torch.no_grad():
        agent1_pos_1_batch, agent2_pos_1_batch = torch.split(agent_pos_1_batch, 2, dim=1)
        o1_1_batch = torch.cat((agent1_pos_1_batch, tasks_state_1_batch), dim=1)
        o2_1_batch = torch.cat((agent2_pos_1_batch, tasks_state_1_batch), dim=1)

    '''if not advantage_function:
        with torch.no_grad():
            agent1_pos_1_batch, agent2_pos_1_batch = torch.split(agent_pos_1_batch, 2, dim=1)
            o1_1_batch = torch.cat((agent1_pos_1_batch, tasks_state_1_batch), dim=1)
            o2_1_batch = torch.cat((agent2_pos_1_batch, tasks_state_1_batch), dim=1)
    else:
        with torch.no_grad():
            agent1_pos_n1_batch, agent2_pos_n1_batch = torch.split(agent_pos_n1_batch, 2, dim=1)
            agent1_pos_1_batch, agent2_pos_1_batch = torch.split(agent_pos_1_batch, 2, dim=1)

            o1_n1_batch = torch.cat((agent1_pos_n1_batch, tasks_state_n1_batch), dim=1)
            o2_n1_batch = torch.cat((agent2_pos_n1_batch, tasks_state_n1_batch), dim=1)
            o1_1_batch = torch.cat((agent1_pos_1_batch, tasks_state_1_batch), dim=1)
            o2_1_batch = torch.cat((agent2_pos_1_batch, tasks_state_1_batch), dim=1)'''

    for update_time in range(max_update_times):

        actor_loss = get_actor_loss_minus_q(agent_pos_1_batch, tasks_state_1_batch, o1_1_batch, o2_1_batch, weights_batch)

        '''if not advantage_function:
            actor_loss = get_actor_loss_minus_q(agent_pos_1_batch, tasks_state_1_batch, o1_1_batch, o2_1_batch, weights_batch)
        else:
            actor_loss = get_actor_loss_advantage_function(agent_pos_n1_batch, agent_pos_1_batch, tasks_state_n1_batch, tasks_state_1_batch,
                                      o1_n1_batch, o2_n1_batch, o1_1_batch, o2_1_batch,
                                      trajectory_reward_batch, weights_batch)'''

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

        (agent_pos_1_batch, agent_action_1_batch, agent_pos_n1_batch, trajectory_reward_batch,
         weights_batch, tasks_state_1_batch, tasks_state_n1_batch) = buffer.sample(batch_size)

        agent_pos_1_batch = agent_pos_1_batch.float().to('cuda')
        agent_action_1_batch = agent_action_1_batch.float().to('cuda')
        agent_pos_n1_batch = agent_pos_n1_batch.float().to('cuda')
        trajectory_reward_batch = trajectory_reward_batch.float().to('cuda')
        weights_batch = weights_batch.float().to('cuda')
        tasks_state_1_batch = tasks_state_1_batch.float().to('cuda')
        tasks_state_n1_batch = tasks_state_n1_batch.float().to('cuda')

    critic_loss = critic_iteration(agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch, trajectory_reward_batch,
                     agent_pos_n1_batch, tasks_state_n1_batch, weights_batch)
    actor_loss = actor_iteration(agent_pos_n1_batch, tasks_state_n1_batch, agent_pos_1_batch, tasks_state_1_batch,
                    trajectory_reward_batch, weights_batch, advantage_function = False)

    return actor_loss, critic_loss


def update_alpha_beta_of_buffer(new_alpha, new_beta):
    buffer.change_alpha(new_alpha)
    buffer.change_beta(new_beta)


def adjust_alpha_beta_sigmoid(t, T, alpha_max, alpha_min, beta_max, beta_min, k=20, p_m=0.5):
    p = t / T  # Normalize step
    alpha = alpha_max - (alpha_max - alpha_min) / (1 + np.exp(-k * (p - p_m)))
    beta = beta_min + (beta_max - beta_min) / (1 + np.exp(-k * (p - p_m)))
    return alpha,beta

def main():
    rounds = 100
    alpha_min = 0.2
    alpha_max = 0.6
    beta_min = 0.4
    beta_max = 0.8
    mini_batch_size = 256  # Mini-Batch 大小

    update_priorities = 0

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

            update_priorities += 1

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
            print(f"10步运行时间: {elapsed_time:.6f} 秒")

            if update_priorities == 1:
                start_time1 = time.time()
                update_priorities_of_buffer()
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
                print(f"优先级更新用时: {elapsed_time1:.6f} 秒")
                update_priorities = 0
            print("\n")

            start_time = time.time()

    path = ["DeepLearning/model/12_25/actor_network_05.pth",
            "DeepLearning/model/12_25/central_critic_network_05.pth"]
    torch.save(actor_network.state_dict(), path[0])
    torch.save(central_critic_network.state_dict(), path[1])




main()