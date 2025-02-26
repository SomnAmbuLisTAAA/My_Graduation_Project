import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
from Neural_network_model import SequentialMultiLayerNN
import gc

agent_pos_dim = 2
agent_action_dim = 2
agent_num = 2

task_pos_dim = 2
task_num = 2

state_space_dim = (agent_pos_dim * agent_num) + (task_pos_dim * task_num) + 1
action_space_dim = agent_action_dim * agent_num
observation_space_dim = agent_pos_dim + (task_pos_dim * task_num) + 1

mini_batch_size = 10000
critic_learning_rate = 0.0001             # 学习率
actor_learning_rate = 0.0001
min_lr = 1e-6
weight_decay = 0.0001             # L2 正则化强度（权重衰减因子）

gamma_power_n = 0.99**50
rounds = 500000
critic_target_network_update_gap = 50
actor_target_network_update_gap = 50

critic_training_loss_memory = []
actor_training_loss_memory = []
critic_training_loss_deque = deque(maxlen=critic_target_network_update_gap)
actor_training_loss_deque = deque(maxlen=actor_target_network_update_gap)

central_critic_network = SequentialMultiLayerNN(state_space_dim + action_space_dim,256,4,1)
central_critic_target_network = SequentialMultiLayerNN(state_space_dim + action_space_dim,256,4,1)
central_critic_network_optimizer = optim.AdamW(central_critic_network.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
central_critic_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(central_critic_network_optimizer, gamma=0.995)

actor_network = SequentialMultiLayerNN(observation_space_dim,256,4,agent_action_dim,0.6,2.0)
actor_target_network = SequentialMultiLayerNN(observation_space_dim,256,4,agent_action_dim,0.6,2.0)
actor_network_optimizer = optim.AdamW(actor_network.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
actor_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer, gamma=0.995)

central_critic_target_network.load_state_dict(central_critic_network.state_dict())
actor_target_network.load_state_dict(actor_network.state_dict())

central_critic_network.to('cuda')
actor_network.to('cuda')

central_critic_target_network.to('cuda')
actor_target_network.to('cuda')

central_critic_network.train()
actor_network.train()

central_critic_target_network.eval()
actor_target_network.eval()

with torch.no_grad():

    path = 'Experience/DataSet_01_09_No2_tensors.pth'
    loaded_tensor = torch.load(path, weights_only=True)

    agent_pos_1_tensor = loaded_tensor['agent_pos_1_tensor']
    agent_pos_n1_tensor = loaded_tensor['agent_pos_n1_tensor']
    tasks_state_1_tensor = loaded_tensor['tasks_state_1_tensor']
    tasks_state_n1_tensor = loaded_tensor['tasks_state_n1_tensor']
    agent_action_1_tensor = loaded_tensor['agent_action_1_tensor']
    trajectory_reward_tensor = loaded_tensor['trajectory_reward_tensor']

    agent_pos_1_tensor = agent_pos_1_tensor.float()
    agent_pos_n1_tensor = agent_pos_n1_tensor.float()
    tasks_state_1_tensor = tasks_state_1_tensor.float()
    tasks_state_n1_tensor = tasks_state_n1_tensor.float()
    agent_action_1_tensor = agent_action_1_tensor.float()
    trajectory_reward_tensor = trajectory_reward_tensor.float()

    # print(trajectory_reward_tensor)

    state_n1_tensor = torch.cat((agent_pos_n1_tensor, tasks_state_n1_tensor), dim=1)
    agent1_pos_n1_tensor, agent2_pos_n1_tensor = torch.split(agent_pos_n1_tensor, 2, dim=1)
    o1_n1_tensor = torch.cat((agent1_pos_n1_tensor, tasks_state_n1_tensor), dim=1)
    o2_n1_tensor = torch.cat((agent2_pos_n1_tensor, tasks_state_n1_tensor), dim=1)

    '''print(state_n1_tensor)
    print(o1_n1_tensor)
    print(o2_n1_tensor)'''

    state_1_tensor = torch.cat((agent_pos_1_tensor, tasks_state_1_tensor), dim=1)
    state_1_action_1_tensor = torch.cat((state_1_tensor, agent_action_1_tensor), dim=1)
    agent1_pos_1_tensor, agent2_pos_1_tensor = torch.split(agent_pos_1_tensor, 2, dim=1)
    o1_1_tensor = torch.cat((agent1_pos_1_tensor, tasks_state_1_tensor), dim=1)
    o2_1_tensor = torch.cat((agent2_pos_1_tensor, tasks_state_1_tensor), dim=1)

    '''print(state_1_tensor)
    print(state_1_action_1_tensor)
    print(o1_1_tensor)
    print(o2_1_tensor)'''

    del (loaded_tensor, agent_pos_1_tensor, agent_pos_n1_tensor, tasks_state_1_tensor, tasks_state_n1_tensor,
         agent_action_1_tensor, agent1_pos_n1_tensor, agent2_pos_n1_tensor, agent1_pos_1_tensor, agent2_pos_1_tensor)
    torch.cuda.empty_cache()
    gc.collect()


    column = state_1_tensor[:, 8]
    for element_to_count in range(4):
        count = (column == element_to_count).sum().item()
        print(f"第 5 列中元素 {element_to_count} 的个数: {count}")

    '''count = (trajectory_reward_tensor >= 0).sum().item()
    print(f"trajectory_reward_tensor中大于0的个数: {count}")'''


start_time = time.time()
for round in range(rounds):

    if (round+1) % (rounds / 500) == 0:
        central_critic_network_scheduler.step()
        actor_network_scheduler.step()

    with torch.no_grad():

        indices = torch.randperm(trajectory_reward_tensor.size(0))[:mini_batch_size]

        trajectory_reward_batch = trajectory_reward_tensor[indices]
        # print(trajectory_reward_batch)

        state_n1_batch = state_n1_tensor[indices]
        o1_n1_batch = o1_n1_tensor[indices]
        o2_n1_batch = o2_n1_tensor[indices]
        '''print(state_n1_batch[0])
        print(o1_n1_batch[0])
        print(o2_n1_batch[0])'''

        state_1_action_1_batch = state_1_action_1_tensor[indices]
        # print(state_1_action_1_batch[0])

        state_1_batch = state_1_tensor[indices]
        o1_1_batch = o1_1_tensor[indices]
        o2_1_batch = o2_1_tensor[indices]
        '''print(state_1_batch[0])
        print(o1_1_batch[0])
        print(o2_1_batch[0])'''

        u_o1_n1_batch = actor_target_network.with_scaled_tanh(o1_n1_batch)
        u_o2_n1_batch = actor_target_network.with_scaled_tanh(o2_n1_batch)
        # print(u_o1_n1_batch[0])
        # print(u_o2_n1_batch[0])

        state_n1_u_state_n1_batch = torch.cat((state_n1_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
        # print(state_n1_u_state_n1_batch[0])
        q_state_n1_u_state_n1_batch = central_critic_target_network(state_n1_u_state_n1_batch)
        # print(q_state_n1_u_state_n1_batch)

        TD_target_batch = trajectory_reward_batch + gamma_power_n * q_state_n1_u_state_n1_batch
        # print(TD_target_batch)

    q_state_1_action_1_batch = central_critic_network(state_1_action_1_batch)
    TD_error_batch = TD_target_batch - q_state_1_action_1_batch
    critic_loss = (0.5 * (TD_error_batch ** 2)).mean()

    u_o1_1_batch = actor_network.with_scaled_tanh(o1_1_batch)
    u_o2_1_batch = actor_network.with_scaled_tanh(o2_1_batch)
    state_1_u_state_1_batch = torch.cat((state_1_batch, u_o1_1_batch, u_o2_1_batch), dim=1)
    # print(state_1_u_state_1_batch)
    q_state_1_u_state_1_batch = central_critic_target_network(state_1_u_state_1_batch)
    actor_loss = (-q_state_1_u_state_1_batch).mean()

    central_critic_network_optimizer.zero_grad()
    actor_network_optimizer.zero_grad()
    critic_loss.backward()
    actor_loss.backward()
    central_critic_network_optimizer.step()
    actor_network_optimizer.step()

    critic_training_loss_deque.append(critic_loss.item())
    actor_training_loss_deque.append(actor_loss.item())

    if len(actor_training_loss_deque) == actor_training_loss_deque.maxlen:
        actor_avg_loss = sum(actor_training_loss_deque) / actor_training_loss_deque.maxlen
        print("actor average loss:", actor_avg_loss)
        actor_training_loss_deque.clear()
        actor_target_network.load_state_dict(actor_network.state_dict())
        actor_training_loss_memory.append(actor_avg_loss)
        for param_group in actor_network_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
            print(f"Actor Current Learning Rate: {param_group['lr']}")

    if len(critic_training_loss_deque) == critic_training_loss_deque.maxlen:
        critic_avg_loss = sum(critic_training_loss_deque) / critic_training_loss_deque.maxlen
        print("critic average loss:", critic_avg_loss)
        critic_training_loss_deque.clear()
        central_critic_target_network.load_state_dict(central_critic_network.state_dict())
        critic_training_loss_memory.append(critic_avg_loss)
        for param_group in central_critic_network_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
            print(f"Critic Current Learning Rate: {param_group['lr']}")

    if (round + 1) % 1000 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"千步运行时间: {elapsed_time:.6f} 秒")
        start_time = time.time()
        print(f"进度：{100 * (round + 1) / rounds}%")

critic_path = "Model/critic_01_09_No2.pth"
torch.save(central_critic_network.state_dict(), critic_path)
actor_path = "Model/actor_01_09_No2.pth"
torch.save(actor_network.state_dict(), actor_path)

x = range(len(critic_training_loss_memory))
plt.plot(x, critic_training_loss_memory)
plt.title('critic_training_loss')
plt.xlabel('training_times')
plt.ylabel('critic_training_loss')
plt.show()

x = range(len(actor_training_loss_memory))
plt.plot(x, actor_training_loss_memory)
plt.title('actor_training_loss')
plt.xlabel('training_times')
plt.ylabel('actor_training_loss')
plt.show()
