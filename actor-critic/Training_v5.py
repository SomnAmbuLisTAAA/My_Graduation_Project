import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import gc

from Interaction_Environment import InteractionEnv
from Neural_network_model import SequentialMultiLayerNN

def soft_update(target_net, main_net, tau=0.005):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


torch.autograd.set_detect_anomaly(True)

device = 'cuda'

agent_init_pos = np.array([ [5.0,5.0],[25.0,5.0] ])
agent_radius = np.array([ [0.2],[0.2] ])
obs_pos = np.array([                        [5.0,15.0],
                                [10.0,10.0],[10.0,15.0],[10.0,20.0],
                     [15.0,5.0],[15.0,10.0],[15.0,15.0],[15.0,20.0],[15.0,25.0],
                                [20.0,10.0],[20.0,15.0],[20.0,20.0],
                                            [25.0,15.0]
                    ])
obs_radius = 1.5 * np.array([       [1.0],
                              [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],[1.0],[1.0],
                              [1.0],[1.0],[1.0],
                                    [1.0]
                           ])
task_pos = np.array([ [7.5,17.5],[17.5,17.5] ])
task_radius = np.array([ [1.0],[1.0] ])
env = InteractionEnv(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,task_pos,task_radius,device='cpu')


agent_pos_dim = 2
agent_action_dim = 2
agent_num = 2
task_pos_dim = 2
task_num = 2
obstacle_num = 13
observation_dim = agent_pos_dim + task_num + 1 + (agent_num - 1)
state_dim = (agent_pos_dim + agent_action_dim) * agent_num + 1

mini_batch_size = 1000
gamma_power_n = 0.95**10
epochs = 100
critic_target_network_update_gap = 1
actor_target_network_update_gap = 1

critic_learning_rate = 1e-4      # 学习率
actor_learning_rate = 1e-6
critic_min_lr = 1e-6
actor_min_lr = 1e-8
weight_decay = 0.0001               # L2 正则化强度（权重衰减因子）

gap_count = 0
actor_update_gap = 10
accumulate_reward_gap = 100
gap_change_flag = 0

break_count = 0

accumulate_reward = None
break_flag = False
task_state = None

reward_memory = []

central_critic_network = SequentialMultiLayerNN(state_dim,128,3,1)
central_critic_target_network = SequentialMultiLayerNN(state_dim,128,3,1)
central_critic_network_optimizer = optim.AdamW(central_critic_network.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
# central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer, mode='min', patience=5, factor=0.5)
central_critic_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(central_critic_network_optimizer, gamma=0.999)

actor_network_1 = SequentialMultiLayerNN(observation_dim, 128,3, agent_action_dim, 0.3, 4.0)
actor_target_network_1 = SequentialMultiLayerNN(observation_dim, 128,3, agent_action_dim, 0.3, 4.0)
actor_network_optimizer_1 = optim.AdamW(actor_network_1.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
# actor_network_scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer_1, mode='max', patience=8, factor=0.8 ,threshold=1e-4, min_lr=1e-8)
actor_network_scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer_1, gamma=0.999)

actor_network_2 = SequentialMultiLayerNN(observation_dim, 128,3, agent_action_dim, 0.3, 4.0)
actor_target_network_2 = SequentialMultiLayerNN(observation_dim, 128,3, agent_action_dim, 0.3, 4.0)
actor_network_optimizer_2 = optim.AdamW(actor_network_2.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
# actor_network_scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer_2, mode='max', patience=8, factor=0.8 ,threshold=1e-4, min_lr=1e-8)
actor_network_scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer_2, gamma=0.999)

central_critic_target_network.load_state_dict(central_critic_network.state_dict())
actor_target_network_1.load_state_dict(actor_network_1.state_dict())
actor_target_network_2.load_state_dict(actor_network_2.state_dict())

central_critic_network.to(device)
central_critic_target_network.to(device)
actor_network_1.to(device)
actor_target_network_1.to(device)
actor_network_2.to(device)
actor_target_network_2.to(device)

central_critic_network.train()
actor_network_1.train()
actor_network_2.train()
central_critic_target_network.eval()
actor_target_network_1.eval()
actor_target_network_2.eval()

with torch.no_grad():

    path = 'Experience/DataSet_02_17_No2_tensors.pth'
    loaded_tensor = torch.load(path, weights_only=True)
    # loaded_tensor = loaded_tensor.to(device)

    trajectory_reward_tensor = loaded_tensor['trajectory_reward_tensor'].to(device)
    o1_1_tensor = loaded_tensor['o1_1_tensor'].to(device)
    o1_n1_tensor = loaded_tensor['o1_n1_tensor'].to(device)
    o2_1_tensor = loaded_tensor['o2_1_tensor'].to(device)
    o2_n1_tensor = loaded_tensor['o2_n1_tensor'].to(device)
    state_1_tensor = loaded_tensor['state_1_tensor'].to(device)
    state_1_without_actions_tensor = loaded_tensor['state_1_without_actions_tensor'].to(device)
    state_n1_without_actions_tensor = loaded_tensor['state_n1_without_actions_tensor'].to(device)

    del (loaded_tensor)
    torch.cuda.empty_cache()
    gc.collect()

    column = o1_1_tensor[:, 4]
    for element_to_count in range(4):
        count = (column == element_to_count).sum().item()
        print(f"第 5 列中元素 {element_to_count} 的个数: {count}")

    dataset = TensorDataset(trajectory_reward_tensor, o1_1_tensor, o1_n1_tensor, o2_1_tensor, o2_n1_tensor,
                            state_1_tensor,state_1_without_actions_tensor, state_n1_without_actions_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        # pin_memory=True  # 固定内存，加速 GPU 访问
    )

round = 0
start_time = time.time()
for epoch in range(epochs):

    if break_flag:
        break

    for trajectory_reward_batch, o1_1_batch, o1_n1_batch, o2_1_batch, o2_n1_batch, state_1_batch, state_1_without_actions_batch, state_n1_without_actions_batch in dataloader:

        if break_flag:
            break

        round += 1

        with torch.no_grad():

            '''trajectory_reward_batch = trajectory_reward_batch.cuda(non_blocking=True)
            o1_1_batch = o1_1_batch.cuda(non_blocking=True)
            o1_n1_batch = o1_n1_batch.cuda(non_blocking=True)
            o2_1_batch = o2_1_batch.cuda(non_blocking=True)
            o2_n1_batch = o2_n1_batch.cuda(non_blocking=True)
            action1_1_batch = action1_1_batch.cuda(non_blocking=True)
            action2_1_batch = action2_1_batch.cuda(non_blocking=True)'''

            u_o1_n1_batch = actor_target_network_1.with_scaled_tanh(o1_n1_batch)
            u_o2_n1_batch = actor_target_network_2.with_scaled_tanh(o2_n1_batch)
            '''print(o1_n1_batch[0])
            print(o2_n1_batch[0])
            print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])'''

            state_n1_batch = torch.cat((state_n1_without_actions_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
            q_state_n1_batch = central_critic_target_network(state_n1_batch)
            TD_target_batch = trajectory_reward_batch + gamma_power_n * q_state_n1_batch.detach()
            '''print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])
            print(state_n1_u_state_n1_batch[0])
            print(q_state_n1_u_state_n1_batch[0])
            print(TD_target_batch[0])'''

        q_state_1_batch = central_critic_network(state_1_batch)
        TD_error_batch = TD_target_batch - q_state_1_batch
        critic_loss = (0.5 * (TD_error_batch ** 2)).mean()

        central_critic_network_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(central_critic_network.parameters(), max_norm=1000.0)
        central_critic_network_optimizer.step()
        # central_critic_network_scheduler.step()

        gap_count += 1

        if gap_count == actor_update_gap:

            gap_count = 0

            u_o1_1_batch = actor_network_1.with_scaled_tanh(o1_1_batch)
            u_o2_1_batch = actor_network_2.with_scaled_tanh(o2_1_batch)

            s1_us1_batch = torch.cat((state_1_without_actions_batch, u_o1_1_batch, u_o2_1_batch), dim=1)
            q_s1_us1_batch = central_critic_target_network(s1_us1_batch)
            actor_loss = -q_s1_us1_batch.mean()

            actor_network_optimizer_1.zero_grad()
            actor_network_optimizer_2.zero_grad()
            actor_loss.backward()
            actor_network_optimizer_1.step()
            actor_network_optimizer_2.step()

        soft_update(actor_target_network_1, actor_network_1, tau=0.01)
        soft_update(actor_target_network_2, actor_network_2, tau=0.01)
        soft_update(central_critic_target_network, central_critic_network, tau=0.01)

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

            accumulate_reward,task_state = env.training_test(actor_network_1,actor_network_2, 1000)
            print(f'accumulate_reward:{accumulate_reward}')
            print(f"task state:{task_state}")

            reward_memory.append(accumulate_reward)

            actor_network_scheduler_1.step()
            actor_network_scheduler_2.step()
            central_critic_network_scheduler.step()

            if task_state == 0 and (epoch+1)/epochs>=0.5:
                break_count += 1

            if break_count == 50:
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

            for param_group in central_critic_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], critic_min_lr)
                print(f"Critic Current Learning Rate: {param_group['lr']}")

            for param_group in actor_network_optimizer_1.param_groups:
                param_group['lr'] = max(param_group['lr'], actor_min_lr)
                print(f"Actor Current Learning Rate: {param_group['lr']}")

    print(f"[epoch/epoches]:[{epoch+1}/{epochs}]", "\n")

critic_path = "Model/critic_02_18_No1.pth"
torch.save(central_critic_network.state_dict(), critic_path)
actor_path = "Model/actor_1_02_18_No1.pth"
torch.save(actor_network_1.state_dict(), actor_path)
actor_path = "Model/actor_2_02_18_No1.pth"
torch.save(actor_network_2.state_dict(), actor_path)

x = range(len(reward_memory))
plt.plot(x, reward_memory)
plt.title('training average return')
plt.xlabel('training_times')
plt.ylabel('average return')
plt.show()