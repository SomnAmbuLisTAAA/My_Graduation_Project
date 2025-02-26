import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import gc
from torch.utils.tensorboard import SummaryWriter


from Central_Critic import CentralCritic
from Interaction_Environment import InteractionEnv
from Neural_network_model import SequentialMultiLayerNN


def soft_update(target_net, main_net, tau=0.005):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


# writer = SummaryWriter(log_dir="runs/0222_1912")

torch.autograd.set_detect_anomaly(True)

device = 'cuda'

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
env = InteractionEnv(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,task_pos,task_radius,device='cpu')


agent_pos_dim = 2
agent_action_dim = 2
agent_num = 2
task_pos_dim = 2
task_num = 4
obstacle_num = 13
# observation_dim = agent_pos_dim + task_num + 1 + (agent_num - 1)
observation_dim = agent_pos_dim + task_num + 1
state_dim = (agent_pos_dim + agent_action_dim) * agent_num + 1

mini_batch_size = 1000
gamma_power_n = 0.95**1
epochs = 2
# critic_target_network_update_gap = 1
# actor_target_network_update_gap = 1

critic_learning_rate = 1e-4       # 学习率
actor_learning_rate = 1e-4
critic_min_lr = 1e-4
actor_min_lr = 1e-4
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
actor2 = SequentialMultiLayerNN(observation_dim,actor_network_size,actor_network_layers,agent_action_dim,0.6,5.0)
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

with torch.no_grad():

    path = 'Experience/DataSet_02_22_No1_tensors.pth'
    loaded_tensor = torch.load(path, weights_only=True)
    # loaded_tensor = loaded_tensor.to(device)

    trajectory_reward_tensor = loaded_tensor['trajectory_reward_tensor'].to('cpu')
    o1_1_tensor = loaded_tensor['o1_1_tensor'].to('cpu')
    o1_n1_tensor = loaded_tensor['o1_n1_tensor'].to('cpu')
    o2_1_tensor = loaded_tensor['o2_1_tensor'].to('cpu')
    o2_n1_tensor = loaded_tensor['o2_n1_tensor'].to('cpu')
    state_1_tensor = loaded_tensor['state_1_tensor'].to('cpu')
    state_1_without_actions_tensor = loaded_tensor['state_1_without_actions_tensor'].to('cpu')
    state_n1_without_actions_tensor = loaded_tensor['state_n1_without_actions_tensor'].to('cpu')

    del (loaded_tensor)
    torch.cuda.empty_cache()
    gc.collect()

    column = o1_1_tensor[:, 6]
    for element_to_count in range(16):
        count = (column == element_to_count).sum().item()
        print(f"第 5 列中元素 {element_to_count} 的个数: {count}")

    dataset = TensorDataset(trajectory_reward_tensor, o1_1_tensor, o1_n1_tensor, o2_1_tensor, o2_n1_tensor,
                            state_1_tensor,state_1_without_actions_tensor, state_n1_without_actions_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        pin_memory=True  # 固定内存，加速 GPU 访问
    )

round = 0
start_time = time.time()
for epoch in range(epochs):

    if break_flag:
        break

    for trajectory_reward_batch, o1_1_batch, o1_n1_batch, o2_1_batch, o2_n1_batch, state_1_batch, state_1_without_actions_batch, state_n1_without_actions_batch in dataloader:

        trajectory_reward_batch = trajectory_reward_batch.to(device)
        o1_1_batch = o1_1_batch.to(device)
        o1_n1_batch = o1_n1_batch.to(device)
        o2_1_batch = o2_1_batch.to(device)
        o2_n1_batch = o2_n1_batch.to(device)
        state_1_batch = state_1_batch.to(device)
        state_1_without_actions_batch = state_1_without_actions_batch.to(device)
        state_n1_without_actions_batch = state_n1_without_actions_batch.to(device)

        if break_flag:
            break

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

        torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=50.0)
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=50.0)

        # torch.nn.utils.clip_grad_norm_(central_critic_network.parameters(), max_norm=0.5)
        # total_grad_critic = torch.sqrt(sum(p.grad.norm()**2 for p in central_critic_network.parameters() if p.grad is not None))
        # writer.add_scalar('Gradient/critic', total_grad_critic, round)

        '''# 记录每一层的梯度直方图
        for name, param in central_critic_network.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'critic_{name}_grad', param.grad, round)  # 记录梯度直方图'''

        # torch.nn.utils.clip_grad_norm_(central_critic_network.parameters(), max_norm=1000.0)
        critic1_optimizer.step()
        critic2_optimizer.step()
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

            torch.nn.utils.clip_grad_norm_(actor1.parameters(), max_norm=5.0)
            # total_grad_actor1 = torch.sqrt(sum(p.grad.norm() ** 2 for p in actor_network1.parameters() if p.grad is not None))
            # writer.add_scalar('Gradient/actor1', total_grad_actor1, round)

            # 记录每一层的梯度直方图
            '''for name, param in actor_network1.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'actor1_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            torch.nn.utils.clip_grad_norm_(actor2.parameters(), max_norm=5.0)
            # total_grad_actor2 = torch.sqrt(sum(p.grad.norm() ** 2 for p in actor_network2.parameters() if p.grad is not None))
            # writer.add_scalar('Gradient/actor2', total_grad_actor2, round)

            # 记录每一层的梯度直方图
            '''for name, param in actor_network2.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'actor2_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            '''# 记录每一层的梯度直方图
            for name, param in central_critic_target_network.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'critic_target_network_{name}_grad', param.grad, round)  # 记录梯度直方图'''

            actor_network_optimizer1.step()
            actor_network_optimizer2.step()

        soft_update(critic_target_network1, critic1, tau=0.001)
        soft_update(critic_target_network2, critic2, tau=0.001)
        soft_update(actor_target_network1, actor1, tau=0.001)
        soft_update(actor_target_network2, actor2, tau=0.001)

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

            accumulate_reward,task_state = env.training_test(actor1,actor2, 1000)
            print(f'accumulate_reward:{accumulate_reward}')
            print(f"task state:{task_state}")

            reward_memory.append(accumulate_reward)

            # central_critic_network_scheduler.step()
            '''actor_network_scheduler1.step()
            actor_network_scheduler2.step()'''

            # actor_network_scheduler.step(accumulate_reward)
            # central_critic_network_scheduler.step()

            if task_state == 0:
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

'''critic_path = "Model/critic_02_20_No2.pth"
torch.save(central_critic_network.state_dict(), critic_path)'''

actor_path = "Model/actor1_02_20_No2.pth"
torch.save(actor1.state_dict(), actor_path)
actor_path = "Model/actor2_02_20_No2.pth"
torch.save(actor2.state_dict(), actor_path)

x = range(len(reward_memory))
plt.plot(x, reward_memory)
plt.title('training average return')
plt.xlabel('training_times')
plt.ylabel('average return')
plt.show()

# writer.close()