import numpy as np
import torch
from mkl_random.mklrand import shuffle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import gc

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from Interaction_Env import Interaction_Env
from Interaction_Environment import RewardEnv
from Neural_network_model import SequentialMultiLayerNN
from Replay_Buffer import ReplayBuffer


def soft_update(target_net, main_net, tau=0.005):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


writer = SummaryWriter(log_dir="runs/0225_2139")
torch.autograd.set_detect_anomaly(True)
device = 'cuda'










'''神经网络初始化'''
agent_pos_dim = 2
agent_action_dim = 2
agent_num = 2
task_pos_dim = 2
task_num = 4
obstacle_num = 13
# observation_dim = agent_pos_dim + task_num + 1 + (agent_num - 1)
# observation_dim = agent_pos_dim + task_num + 1
observation_dim = agent_pos_dim + 1
state_dim = (agent_pos_dim + agent_action_dim) * agent_num + 1

mini_batch_size = 1000
gamma_power_n = 0.95**5
epochs = 2500
# critic_target_network_update_gap = 1
# actor_target_network_update_gap = 1

critic_learning_rate = 1e-4       # 学习率
actor_learning_rate = 1e-6
critic_min_lr = 1e-4
actor_min_lr = 1e-6
weight_decay = 0.0001               # L2 正则化强度（权重衰减因子）

gap_count = 0
actor_update_gap = 10
accumulate_reward_gap = 100

accumulate_reward = None
break_flag = False
task_state = None

reward_memory = []
# critic_training_loss_deque = deque(maxlen=100)
# actor_training_loss_deque = deque(maxlen=actor_target_network_update_gap)

# central_critic_network = CentralCritic(5,4)
# central_critic_target_network = CentralCritic(5,4)
# central_critic_network_optimizer = optim.AdamW(central_critic_network.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
# central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer, mode='min', patience=5, factor=0.5)
# central_critic_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(central_critic_network_optimizer, gamma=0.995)

critic_network_size = 256
critic_network_layers = 4

critic1 = SequentialMultiLayerNN(state_dim,critic_network_size,critic_network_layers,1)
critic2 = SequentialMultiLayerNN(state_dim,critic_network_size,critic_network_layers,1)
critic_target_network1 = SequentialMultiLayerNN(state_dim,critic_network_size,critic_network_layers,1)
critic_target_network2 = SequentialMultiLayerNN(state_dim,critic_network_size,critic_network_layers,1)
critic1_optimizer = optim.AdamW(critic1.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
critic2_optimizer = optim.AdamW(critic2.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)

actor_network_size = 256
actor_network_layers = 4

actor1 = SequentialMultiLayerNN(observation_dim,actor_network_size,actor_network_layers,agent_action_dim,0.6,2.5)
actor_target_network1 = SequentialMultiLayerNN(observation_dim,actor_network_size,actor_network_layers,agent_action_dim,0.6,2.5)
actor_network_optimizer1 = optim.AdamW(actor1.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
actor2 = SequentialMultiLayerNN(observation_dim,actor_network_size,actor_network_layers,agent_action_dim,0.6,2.5)
actor_target_network2 = SequentialMultiLayerNN(observation_dim,actor_network_size,actor_network_layers,agent_action_dim,0.6,2.5)
actor_network_optimizer2 = optim.AdamW(actor2.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)

'''actor_network_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer1, gamma=0.995)
actor_network_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer2, gamma=0.995)'''

'''actor_network_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer1, mode='max', patience=5, factor=0.8 ,threshold=1e-4, min_lr=1e-8)
actor_network_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer2, mode='max', patience=5, factor=0.8 ,threshold=1e-4, min_lr=1e-8)'''
# actor_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer, gamma=0.95)

# central_critic_target_network.load_state_dict(central_critic_network.state_dict())
critic_target_network1.load_state_dict(critic1.state_dict())
critic_target_network2.load_state_dict(critic2.state_dict())

actor_target_network1.load_state_dict(actor1.state_dict())
# actor2.load_state_dict(actor1.state_dict())
actor_target_network2.load_state_dict(actor2.state_dict())

'''central_critic_network.to(device)
central_critic_target_network.to(device)'''

critic1.to(device)
critic_target_network1.to(device)
critic2.to(device)
critic_target_network2.to(device)

actor1.to(device)
actor_target_network1.to(device)
actor2.to(device)
actor_target_network2.to(device)

# central_critic_network.train()
actor1.train()
actor2.train()
critic1.train()
critic2.train()

# central_critic_target_network.eval()
actor_target_network1.eval()
actor_target_network2.eval()
critic_target_network1.eval()
critic_target_network2.eval()
'''神经网络初始化-end'''










'''交互环境初始化'''
agent_init_pos = np.array([ [5.0,5.0],[25.0,5.0] ])
agent_radius = np.array([ [0.1],[0.1] ])
obs_pos = np.array([                        [5.0,15.0],
                                [10.0,10.0],[10.0,15.0],[10.0,20.0],
                     [15.0,5.0],[15.0,10.0],[15.0,15.0],[15.0,20.0],[15.0,25.0],
                                [20.0,10.0],[20.0,15.0],[20.0,20.0],
                                            [25.0,15.0]
                    ])
obs_radius = 0.8 * np.array([       [1.0],
                              [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],[1.0],[1.0],
                              [1.0],[1.0],[1.0],
                                    [1.0]
                           ])
task_pos = np.array([ [5.0,25.0],[7.5,17.5],[17.5,17.5],[22.5,12.5] ])
task_radius = np.array([ [0.5],[0.5],[0.5],[0.5] ])
env = Interaction_Env(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,task_pos,task_radius,device='cpu')
reward_env = RewardEnv(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,task_pos,task_radius,device='cpu')
'''交互环境初始化-end'''










'''经验回放池初始化'''
buffer_max_size = 100000
buffer = ReplayBuffer(max_size=buffer_max_size)

def buffer_init():

    print("初始化开始")

    init_path = Path("Experience/Init_Data.npz")

    if init_path.is_file():
        buffer.load_data(init_path.as_posix())

    else:
        line = 0

        trajectory_reward_matrix = np.empty((buffer_max_size, 2), dtype=np.float32)
        o1_1_matrix = np.empty((buffer_max_size, 3), dtype=np.float32)
        o2_1_matrix = np.empty((buffer_max_size, 3), dtype=np.float32)
        o1_n1_matrix = np.empty((buffer_max_size, 3), dtype=np.float32)
        o2_n1_matrix = np.empty((buffer_max_size, 3), dtype=np.float32)
        state_1_matrix = np.empty((buffer_max_size, 9), dtype=np.float32)
        state_1_without_actions_matrix = np.empty((buffer_max_size, 5), dtype=np.float32)
        state_n1_without_actions_matrix = np.empty((buffer_max_size, 5), dtype=np.float32)

        while line != buffer_max_size:

            observation_1, observation_n1, state_1, state_1_without_actions, state_n1_without_actions, trajectory_reward = env.step(
                mode='explore')

            if not observation_1.size == 0:
                trajectory_reward_matrix[line] = trajectory_reward
                o1_1_matrix[line] = observation_1[0]
                o2_1_matrix[line] = observation_1[1]
                o1_n1_matrix[line] = observation_n1[0]
                o2_n1_matrix[line] = observation_n1[1]
                state_1_matrix[line] = state_1
                state_1_without_actions_matrix[line] = state_1_without_actions
                state_n1_without_actions_matrix[line] = state_n1_without_actions
                line += 1

                if observation_1[0][2] != 15:
                    env.reset_task()

                '''if observation_1[0][2] not in [1, 3, 5, 7, 9, 11, 13, 15] and 0 < (line + 1) / buffer_max_size <= 0.25:
                    env.reset_task([3])

                if observation_1[0][2] not in [2, 3, 6, 7, 10, 11, 14, 15] and 0.25 < (line + 1) / buffer_max_size <= 0.50:
                    env.reset_task([2])

                if observation_1[0][2] not in [4, 5, 6, 7, 12, 13, 14, 15] and 0.50 < (line + 1) / buffer_max_size <= 0.75:
                    env.reset_task([1])

                if observation_1[0][2] not in [8, 9, 10, 11, 12, 13, 14, 15] and 0.75 < (line + 1) / buffer_max_size:
                    env.reset_task([0])'''

            if (line + 1) % 1000 == 0:
                print(f"初始化进度:{(100 * (line + 1) / buffer_max_size):.2f}%")

        buffer.update_data(trajectory_reward_matrix, o1_1_matrix, o2_1_matrix, o1_n1_matrix,
                           o2_n1_matrix, state_1_matrix, state_1_without_actions_matrix,
                           state_n1_without_actions_matrix)
        buffer.save_data(init_path.as_posix())

    buffer.shuffle_data()


def buffer_update(update_size):

    env.load_actor(actor1,actor2)
    env.reset()

    line = 0

    trajectory_reward_matrix = np.empty((update_size, 2), dtype=np.float32)
    o1_1_matrix = np.empty((update_size, 3), dtype=np.float32)
    o2_1_matrix = np.empty((update_size, 3), dtype=np.float32)
    o1_n1_matrix = np.empty((update_size, 3), dtype=np.float32)
    o2_n1_matrix = np.empty((update_size, 3), dtype=np.float32)
    state_1_matrix = np.empty((update_size, 9), dtype=np.float32)
    state_1_without_actions_matrix = np.empty((update_size, 5), dtype=np.float32)
    state_n1_without_actions_matrix = np.empty((update_size, 5), dtype=np.float32)

    while line != update_size:

        observation_1, observation_n1, state_1, state_1_without_actions, state_n1_without_actions, trajectory_reward = env.step(mode='default')

        if not observation_1.size == 0:
            trajectory_reward_matrix[line] = trajectory_reward
            o1_1_matrix[line] = observation_1[0]
            o2_1_matrix[line] = observation_1[1]
            o1_n1_matrix[line] = observation_n1[0]
            o2_n1_matrix[line] = observation_n1[1]
            state_1_matrix[line] = state_1
            state_1_without_actions_matrix[line] = state_1_without_actions
            state_n1_without_actions_matrix[line] = state_n1_without_actions
            line += 1

            if observation_1[0][2] not in [1, 3, 5, 7, 9, 11, 13, 15] and 0 < (line + 1)/ update_size <= 0.25:
                env.reset_task([3])

            if observation_1[0][2] not in [2, 3, 6, 7, 10, 11, 14, 15] and 0.25 < (line + 1) / update_size <= 0.50:
                env.reset_task([2])

            if observation_1[0][2] not in [4, 5, 6, 7, 12, 13, 14, 15] and 0.50 < (line + 1) / update_size <= 0.75:
                env.reset_task([1])

            if observation_1[0][2] not in [8, 9, 10, 11, 12, 13, 14, 15] and 0.75 < (line + 1) / update_size:
                env.reset_task([0])

            '''if (line + 1) % 1000 == 0:
                print(f"更新进度:{(100 * (line + 1) / update_size):.2f}%")'''

    buffer.update_data(trajectory_reward_matrix, o1_1_matrix, o2_1_matrix, o1_n1_matrix,
                       o2_n1_matrix, state_1_matrix, state_1_without_actions_matrix,
                       state_n1_without_actions_matrix, shuffle=True)
    # buffer.shuffle_data()

    print("经验池更新结束")

buffer_init()
# sampler = RandomSampler(buffer)
# dataloader = DataLoader(buffer, batch_size=mini_batch_size, pin_memory=True, sampler=sampler)
dataloader = DataLoader(buffer, batch_size=mini_batch_size, pin_memory=True, shuffle=True)
'''经验回放池初始化-end'''










round = 0
start_time = time.time()
for epoch in range(epochs):

    if break_flag:
        break

    for batch in dataloader:

        if break_flag:
            break

        trajectory_reward_batch = batch["trajectory_reward"]
        o1_1_batch = batch["o1_1"]
        o1_n1_batch = batch["o1_n1"]
        o2_1_batch = batch["o2_1"]
        o2_n1_batch = batch["o2_n1"]
        state_1_batch = batch["state_1"]
        state_1_without_actions_batch = batch["state_1_without_actions"]
        state_n1_without_actions_batch = batch["state_n1_without_actions"]

        trajectory_reward_batch = trajectory_reward_batch.to(device)
        o1_1_batch = o1_1_batch.to(device)
        o1_n1_batch = o1_n1_batch.to(device)
        o2_1_batch = o2_1_batch.to(device)
        o2_n1_batch = o2_n1_batch.to(device)
        state_1_batch = state_1_batch.to(device)
        state_1_without_actions_batch = state_1_without_actions_batch.to(device)
        state_n1_without_actions_batch = state_n1_without_actions_batch.to(device)

        round += 1

        with (torch.no_grad()):

            '''trajectory_reward_batch = trajectory_reward_batch.cuda(non_blocking=True)
            o1_1_batch = o1_1_batch.cuda(non_blocking=True)
            o1_n1_batch = o1_n1_batch.cuda(non_blocking=True)
            o2_1_batch = o2_1_batch.cuda(non_blocking=True)
            o2_n1_batch = o2_n1_batch.cuda(non_blocking=True)
            action1_1_batch = action1_1_batch.cuda(non_blocking=True)
            action2_1_batch = action2_1_batch.cuda(non_blocking=True)'''

            '''print(trajectory_reward_batch)
            print(o1_1_batch)
            print(o2_1_batch)'''

            u_o1_n1_batch = actor_target_network1.with_scaled_tanh(o1_n1_batch)
            u_o2_n1_batch = actor_target_network2.with_scaled_tanh(o2_n1_batch)
            '''print(o1_n1_batch[0])
            print(o2_n1_batch[0])
            print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])'''

            state_n1_batch = torch.cat((state_n1_without_actions_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
            q1_state_n1_batch = critic_target_network1(state_n1_batch)
            q2_state_n1_batch = critic_target_network2(state_n1_batch.detach())

            q1_target = trajectory_reward_batch[:, 0].view(-1,1) + gamma_power_n * q1_state_n1_batch
            q2_target = trajectory_reward_batch[:, 1].view(-1,1) + gamma_power_n * q2_state_n1_batch

            '''print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])
            print(state_n1_u_state_n1_batch[0])
            print(q_state_n1_u_state_n1_batch[0])
            print(TD_target_batch[0])'''

        q1_current = critic1(state_1_batch)
        q2_current = critic2(state_1_batch.detach())

        critic1_loss = 0.5 * ((q1_target - q1_current)**2).mean()
        critic2_loss = 0.5 * ((q2_target - q2_current)**2).mean()

        critic1_optimizer.zero_grad()
        critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()

        torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=5.0)

        '''torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=2000.0)
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=2000.0)'''

        '''total_grad_critic = torch.sqrt(sum(p.grad.norm()**2 for p in critic1.parameters() if p.grad is not None))
        writer.add_scalar('Gradient/critic1', total_grad_critic, round)
        total_grad_critic = torch.sqrt(sum(p.grad.norm()**2 for p in critic2.parameters() if p.grad is not None))
        writer.add_scalar('Gradient/critic2', total_grad_critic, round)'''

        '''# 记录每一层的梯度直方图
        for name, param in central_critic_network.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'critic_{name}_grad', param.grad, round)  # 记录梯度直方图'''

        # torch.nn.utils.clip_grad_norm_(central_critic_network.parameters(), max_norm=1000.0)
        critic1_optimizer.step()
        critic2_optimizer.step()
        soft_update(critic_target_network1, critic1, tau=0.05)
        soft_update(critic_target_network2, critic2, tau=0.05)
        # critic_training_loss_deque.append(critic_loss.item())
        # central_critic_network_scheduler.step()

        gap_count += 1

        if gap_count == actor_update_gap:

            gap_count = 0

            u_o1_1_batch = actor1.with_scaled_tanh(o1_1_batch)
            u_o2_1_batch = actor2.with_scaled_tanh(o2_1_batch)

            agent1_s1_us1_batch = torch.cat((state_1_without_actions_batch.detach(), u_o1_1_batch, u_o2_1_batch.detach()), dim=1)
            agent2_s1_us1_batch = torch.cat((state_1_without_actions_batch.detach(), u_o1_1_batch.detach(), u_o2_1_batch), dim=1)

            q1_s1_us1_batch = critic_target_network1(agent1_s1_us1_batch)
            q2_s1_us1_batch = critic_target_network2(agent2_s1_us1_batch)

            actor1_loss = -1 * q1_s1_us1_batch.mean()
            actor2_loss = -1 * q2_s1_us1_batch.mean()

            actor_network_optimizer1.zero_grad()
            actor_network_optimizer2.zero_grad()
            actor1_loss.backward()
            actor2_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor1.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(actor2.parameters(), max_norm=0.5)

            actor_network_optimizer1.step()
            actor_network_optimizer2.step()

            soft_update(actor_target_network1, actor1, tau=0.05)
            soft_update(actor_target_network2, actor2, tau=0.05)

            '''total_grad_actor1 = torch.sqrt(sum(p.grad.norm() ** 2 for p in actor1.parameters() if p.grad is not None))
            writer.add_scalar('Gradient/actor1', total_grad_actor1, round)
            total_grad_actor2 = torch.sqrt(sum(p.grad.norm() ** 2 for p in actor2.parameters() if p.grad is not None))
            writer.add_scalar('Gradient/actor2', total_grad_actor2, round)'''

            # 记录每一层的梯度直方图
            '''for name, param in actor_network1.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'actor1_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            # 记录每一层的梯度直方图
            '''for name, param in actor_network2.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'actor2_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            '''# 记录每一层的梯度直方图
            for name, param in central_critic_target_network.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'critic_target_network_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            # buffer_update(10000)

        '''if epoch+1<=1:
            central_critic_target_network.load_state_dict(central_critic_network.state_dict())
            actor_target_network1.load_state_dict(actor_network1.state_dict())
            actor_target_network2.load_state_dict(actor_network2.state_dict())'''

        '''else:
            soft_update(actor_target_network1, actor_network1, tau=0.005)
            soft_update(actor_target_network2, actor_network2, tau=0.005)
            soft_update(central_critic_target_network, central_critic_network, tau=0.005)'''

        '''for param_group in central_critic_network_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], critic_min_lr)
            print(f"Critic Current Learning Rate: {param_group['lr']}")

        for param_group in actor_network_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], actor_min_lr)
            print(f"Actor Current Learning Rate: {param_group['lr']}")'''

        if (round+1)%accumulate_reward_gap == 0:

            '''for param_group in actor_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], actor_min_lr)
                print(f"Actor Current Learning Rate: {param_group['lr']}")'''

            '''critic_avg_loss = sum(critic_training_loss_deque) / len(critic_training_loss_deque)
            print("critic average loss:", critic_avg_loss)
            critic_training_loss_deque.clear()'''

            accumulate_reward,task_state = reward_env.training_test(actor1,actor2, 1000)
            print(f'accumulate_reward:{accumulate_reward}')
            print(f"task state:{task_state}")
            reward_memory.append(accumulate_reward)
            writer.add_scalar('Gradient/average_return', accumulate_reward, round)
            buffer_update(10000)

            # central_critic_network_scheduler.step()
            '''actor_network_scheduler1.step()
            actor_network_scheduler2.step()'''

            # actor_network_scheduler.step(accumulate_reward)
            # central_critic_network_scheduler.step()

            if task_state == 0 and epoch+1 > 2000:
                break_flag = True



        '''if len(critic_training_loss_deque) == critic_training_loss_deque.maxlen:
            critic_avg_loss = sum(critic_training_loss_deque) / critic_training_loss_deque.maxlen
            print("critic average loss:", critic_avg_loss)
            critic_training_loss_deque.clear()
            critic_training_loss_memory.append(critic_avg_loss)
            # central_critic_network_scheduler.step()

            for param_group in central_critic_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], critic_min_lr)
                print(f"Critic Current Learning Rate: {param_group['lr']}")
                # central_critic_network_scheduler.step()
                # actor_network_scheduler.step()

        if len(actor_training_loss_deque) == actor_training_loss_deque.maxlen:
            actor_avg_loss = sum(actor_training_loss_deque) / actor_training_loss_deque.maxlen
            print("actor average loss:", actor_avg_loss)
            actor_training_loss_deque.clear()
            actor_training_loss_memory.append(actor_avg_loss)

            for param_group in actor_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], actor_min_lr)
                print(f"Actor Current Learning Rate: {param_group['lr']}")

            accumulate_reward = env.training_test(actor_network,100)
            # central_critic_network_scheduler.step()
            actor_network_scheduler.step()
            print(f'accumulate_reward:{accumulate_reward}', '\n')'''

        if round % 1000 == 0:

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"千步运行时间: {elapsed_time:.6f} 秒")
            start_time = time.time()

            for param_group in critic1_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], critic_min_lr)
                print(f"Critic Current Learning Rate: {param_group['lr']}")

            for param_group in actor_network_optimizer1.param_groups:
                param_group['lr'] = max(param_group['lr'], actor_min_lr)
                print(f"Actor Current Learning Rate: {param_group['lr']}")

    print(f"[epoch/epoches]:[{epoch+1}/{epochs}]", "\n")


actor_path = "Model/actor1_02_26_No1.pth"
torch.save(actor1.state_dict(), actor_path)
actor_path = "Model/actor2_02_26_No1.pth"
torch.save(actor2.state_dict(), actor_path)

x = range(len(reward_memory))
plt.plot(x, reward_memory)
plt.title('training average return')
plt.xlabel('training_times')
plt.ylabel('average return')
plt.show()

writer.close()