import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
from NN_model import SequentialMultiLayerNN
import gc

mini_batch_size = 10000
learning_rate = 0.001             # 学习率
weight_decay = 0.0001              # L2 正则化强度（权重衰减因子）
patience = 20                     # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5                      # 学习率衰减参数
max_iteration_times = 1
relative_threshold = 1e-6

min_lr = 1e-6

gamma_power_n = 0.99**20
rounds = 100000

global_loss_mean = 0
training_loss_memory = []
several_training_loss = deque(maxlen=50)

adjust_LR = 0

Deep_Q_Network = SequentialMultiLayerNN(4,256,4,1)
'''Deep_Q_Network：主网络'''

Deep_Q_Network_optimizer = optim.AdamW(Deep_Q_Network.parameters(), lr=learning_rate, weight_decay=weight_decay)
Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ExponentialLR(Deep_Q_Network_optimizer, gamma=0.995)
# Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min', patience=patience, threshold=1e-5, factor=factor, min_lr=1e-6)

Deep_Q_Target_Network = SequentialMultiLayerNN(4,256,4,1)
'''Deep_Q_Target_Network：目标网络'''

Deep_Q_Target_Network.load_state_dict(Deep_Q_Network.state_dict())

Deep_Q_Network.to('cuda')
Deep_Q_Target_Network.to('cuda')

Deep_Q_Network.train()
Deep_Q_Target_Network.eval()

with torch.no_grad():
    path = 'Experience/DataSet_01_05_No2_tensors.pth'
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

    all_action = torch.arange(1, 10).view(-1, 1)
    all_action_batch = all_action.repeat(mini_batch_size, 1).float()
    all_action_batch = all_action_batch.to('cuda')

    state_1_action_1_tensor = torch.cat((agent_pos_1_tensor, tasks_state_1_tensor, agent_action_1_tensor), dim=1)

    state_n1_tensor = torch.cat((agent_pos_n1_tensor, tasks_state_n1_tensor), dim=1)
    state_n1_tensor = state_n1_tensor.to('cuda')

    del (loaded_tensor, agent_pos_1_tensor, agent_pos_n1_tensor, tasks_state_1_tensor, tasks_state_n1_tensor,
         agent_action_1_tensor, all_action)
    torch.cuda.empty_cache()
    gc.collect()

start_time = time.time()
for round in range(rounds):

    '''for param_group in Deep_Q_Network_optimizer.param_groups:
        param_group['lr'] = learning_rate'''

    if (round+1) % 200 == 0:
        Deep_Q_Network_scheduler.step()
        # Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] *= 0.997

    '''if (round+1) == 7000:
        for param_group in Deep_Q_Network_optimizer.param_groups:
            param_group['lr'] *= 0.5'''

    with torch.no_grad():

        indices = torch.randperm(trajectory_reward_tensor.size(0))[:mini_batch_size]

        trajectory_reward_batch = trajectory_reward_tensor[indices]
        state_n1_batch = state_n1_tensor[indices]
        state_1_action_1_batch = state_1_action_1_tensor[indices]

        state_n1_batch = state_n1_batch.repeat_interleave(9, dim=0)
        state_n1_all_action_batch = torch.cat((state_n1_batch, all_action_batch), dim=1)
        # print(state_n1_all_action_batch)

        q_value_state_n1_all_action_batch = Deep_Q_Target_Network(state_n1_all_action_batch)
        # print(q_value_state_n1_all_action_batch)
        reshaped_q_value_batch = q_value_state_n1_all_action_batch.view(-1, 9)
        max_q_values, _ = reshaped_q_value_batch.max(dim=1)
        max_q_s_n1 = max_q_values.view(-1, 1)

        TD_target_batch = trajectory_reward_batch + gamma_power_n * max_q_s_n1
        '''print(trajectory_reward_batch)
        print(max_q_s_n1)
        print(state_1_action_1_batch)'''

    for update_time in range(max_iteration_times):
        # print(state_1_action_1)
        TD_error = TD_target_batch - Deep_Q_Network(state_1_action_1_batch)
        # print(TD_error)
        loss = 0.5 * (TD_error ** 2)
        # print(loss)
        loss = loss.mean()
        # print(loss)

        Deep_Q_Network_optimizer.zero_grad()
        loss.backward()
        Deep_Q_Network_optimizer.step()

        several_training_loss.append(loss.item())


    if len(several_training_loss) == several_training_loss.maxlen:
        avg_loss = sum(several_training_loss)/several_training_loss.maxlen
        print("batch average loss:", avg_loss)
        several_training_loss.clear()
        Deep_Q_Target_Network.load_state_dict(Deep_Q_Network.state_dict())
        # Deep_Q_Network_scheduler.step()
        training_loss_memory.append(avg_loss)

        # Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] *= 0.98
        for param_group in Deep_Q_Network_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
            print(f"Current Learning Rate: {param_group['lr']}")

    if (round + 1) % 1000 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"千步运行时间: {elapsed_time:.6f} 秒")
        start_time = time.time()
        print(f"进度：{100 * (round + 1) / rounds}%")

path = "model/Deep_Q_network_01_05_No2.pth"
torch.save(Deep_Q_Network.state_dict(), path)

x = range(len(training_loss_memory))
plt.plot(x, training_loss_memory)
plt.title('training_loss')
plt.xlabel('training_times')
plt.ylabel('training_loss')
plt.show()
