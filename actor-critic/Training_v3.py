import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
from Neural_network_model import SequentialMultiLayerNN
import gc
torch.autograd.set_detect_anomaly(True)

device = 'cuda'

agent_pos_dim = 2
agent_action_dim = 2
agent_num = 2
task_pos_dim = 2
task_num = 2
obstacle_num = 13
observation_dim = agent_pos_dim + task_num + 1 + agent_num + obstacle_num
state_dim = (observation_dim + agent_action_dim) * agent_num

mini_batch_size = 1000
gamma_power_n = 0.99**50
epochs = 100
critic_target_network_update_gap = 20
actor_target_network_update_gap = 20

critic_learning_rate = 0.01       # 学习率
actor_learning_rate = 0.0001
critic_min_lr = 1e-6
actor_min_lr = 1e-6
weight_decay = 0.0001               # L2 正则化强度（权重衰减因子）

critic_training_loss_memory = []
actor_training_loss_memory = []
critic_training_loss_deque = deque(maxlen=critic_target_network_update_gap)
actor_training_loss_deque = deque(maxlen=actor_target_network_update_gap)

central_critic_network = SequentialMultiLayerNN(state_dim,256,4,1)
central_critic_target_network = SequentialMultiLayerNN(state_dim,256,4,1)
central_critic_network_optimizer = optim.AdamW(central_critic_network.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
central_critic_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(central_critic_network_optimizer, mode='min', patience=3, factor=0.5)
# central_critic_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(central_critic_network_optimizer, gamma=0.995)

actor_network = SequentialMultiLayerNN(observation_dim,256,4,agent_action_dim,0.6,2.0)
actor_target_network = SequentialMultiLayerNN(observation_dim,256,4,agent_action_dim,0.6,2.0)
actor_network_optimizer = optim.AdamW(actor_network.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
# actor_network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_network_optimizer, mode='min', patience=5, factor=0.8)
# actor_network_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_network_optimizer, gamma=0.97)

central_critic_target_network.load_state_dict(central_critic_network.state_dict())
actor_target_network.load_state_dict(actor_network.state_dict())

central_critic_network.to(device)
central_critic_target_network.to(device)
actor_network.to(device)
actor_target_network.to(device)

central_critic_network.train()
actor_network.train()
central_critic_target_network.eval()
actor_target_network.eval()

with torch.no_grad():

    path = 'Experience/DataSet_02_09_No2_tensors.pth'
    loaded_tensor = torch.load(path, weights_only=True)
    # loaded_tensor = loaded_tensor.to(device)

    trajectory_reward_tensor = loaded_tensor['trajectory_reward_tensor'].to(device)
    o1_1_tensor = loaded_tensor['o1_1_tensor'].to(device)
    o1_n1_tensor = loaded_tensor['o1_n1_tensor'].to(device)
    o2_1_tensor = loaded_tensor['o2_1_tensor'].to(device)
    o2_n1_tensor = loaded_tensor['o2_n1_tensor'].to(device)
    action1_1_tensor = loaded_tensor['action1_1_tensor'].to(device)
    action2_1_tensor = loaded_tensor['action2_1_tensor'].to(device)

    del (loaded_tensor)
    torch.cuda.empty_cache()
    gc.collect()

    column = o1_1_tensor[:, 4]
    for element_to_count in range(4):
        count = (column == element_to_count).sum().item()
        print(f"第 5 列中元素 {element_to_count} 的个数: {count}")

    dataset = TensorDataset(trajectory_reward_tensor, o1_1_tensor, o1_n1_tensor, o2_1_tensor, o2_n1_tensor,
                            action1_1_tensor, action2_1_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        # pin_memory=True  # 固定内存，加速 GPU 访问
    )

round = 0
start_time = time.time()
for epoch in range(epochs):

    for trajectory_reward_batch, o1_1_batch, o1_n1_batch, o2_1_batch, o2_n1_batch, action1_1_batch, action2_1_batch in dataloader:

        round += 1

        with torch.no_grad():

            '''trajectory_reward_batch = trajectory_reward_batch.cuda(non_blocking=True)
            o1_1_batch = o1_1_batch.cuda(non_blocking=True)
            o1_n1_batch = o1_n1_batch.cuda(non_blocking=True)
            o2_1_batch = o2_1_batch.cuda(non_blocking=True)
            o2_n1_batch = o2_n1_batch.cuda(non_blocking=True)
            action1_1_batch = action1_1_batch.cuda(non_blocking=True)
            action2_1_batch = action2_1_batch.cuda(non_blocking=True)'''

            u_o1_n1_batch = actor_target_network.with_scaled_tanh(o1_n1_batch)
            u_o2_n1_batch = actor_target_network.with_scaled_tanh(o2_n1_batch)
            '''print(o1_n1_batch[0])
            print(o2_n1_batch[0])
            print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])'''

            state_n1_batch = torch.cat((o1_n1_batch, o2_n1_batch, u_o1_n1_batch, u_o2_n1_batch), dim=1)
            q_state_n1_batch = central_critic_target_network(state_n1_batch)
            TD_target_batch = trajectory_reward_batch + gamma_power_n * q_state_n1_batch.detach()
            '''print(u_o1_n1_batch[0])
            print(u_o2_n1_batch[0])
            print(state_n1_u_state_n1_batch[0])
            print(q_state_n1_u_state_n1_batch[0])
            print(TD_target_batch[0])'''

            state_1_batch = torch.cat((o1_1_batch, o2_1_batch, action1_1_batch, action2_1_batch), dim=1)
            '''print(action1_1_batch[0])
            print(action1_1_batch[0])
            print(o1_1_batch[0])
            print(o2_1_batch[0])'''

        q_state_1_batch = central_critic_network(state_1_batch)
        TD_error_batch = TD_target_batch - q_state_1_batch
        critic_loss = (0.5 * (TD_error_batch ** 2)).mean()

        central_critic_network_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(central_critic_network.parameters(), max_norm=1000.0)
        central_critic_network_optimizer.step()
        critic_training_loss_deque.append(critic_loss.item())

        u_o1_1_batch = actor_network.with_scaled_tanh(o1_1_batch)
        u_o2_1_batch = actor_network.with_scaled_tanh(o2_1_batch)

        s1_us1_batch = torch.cat((o1_1_batch, o2_1_batch, u_o1_1_batch, u_o2_1_batch), dim=1)
        q_s1_us1_batch = central_critic_target_network(s1_us1_batch)
        actor_loss = -q_s1_us1_batch.mean()

        actor_network_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=10.0)
        actor_network_optimizer.step()
        actor_training_loss_deque.append(actor_loss.item())

        if len(actor_training_loss_deque) == actor_training_loss_deque.maxlen:
            actor_avg_loss = sum(actor_training_loss_deque) / actor_training_loss_deque.maxlen
            print("actor average loss:", actor_avg_loss)
            actor_training_loss_deque.clear()
            actor_target_network.load_state_dict(actor_network.state_dict())
            actor_training_loss_memory.append(actor_avg_loss)

        if len(critic_training_loss_deque) == critic_training_loss_deque.maxlen:
            critic_avg_loss = sum(critic_training_loss_deque) / critic_training_loss_deque.maxlen
            print("critic average loss:", critic_avg_loss)
            critic_training_loss_deque.clear()
            central_critic_target_network.load_state_dict(central_critic_network.state_dict())
            critic_training_loss_memory.append(critic_avg_loss)

        if round % 100 == 0:

            central_critic_network_scheduler.step(critic_avg_loss)
            '''if critic_avg_loss <= 4000:
                actor_network_scheduler.step()'''

            for param_group in central_critic_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], critic_min_lr)
                print(f"Critic Current Learning Rate: {param_group['lr']}")

            for param_group in actor_network_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], actor_min_lr)
                print(f"Actor Current Learning Rate: {param_group['lr']}", "\n")

        if round % 1000 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"千步运行时间: {elapsed_time:.6f} 秒")
            start_time = time.time()

critic_path = "Model/critic_02_10_No2.pth"
torch.save(central_critic_network.state_dict(), critic_path)
actor_path = "Model/actor_02_10_No2.pth"
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