import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
from NN_model import SequentialMultiLayerNN

mini_batch_size = 64
learning_rate = 0.1      # 学习率
weight_decay = 0.01      # L2 正则化强度（权重衰减因子）
patience = 3             # 学习率衰减参数 如果经过patience个周期的验证集损失或其他指标都没有显著降低，学习率就会乘factor
factor = 0.5             # 学习率衰减参数
max_iteration_times = 1
relative_threshold = 1e-4

gamma_power_n = 0.99**10
rounds = 10000

training_loss_memory = []
several_training_loss = deque(maxlen=100)

Deep_Q_Network = SequentialMultiLayerNN(4,128,3,1)
'''Deep_Q_Network：主网络'''
Deep_Q_Network_optimizer = optim.Adam(Deep_Q_Network.parameters(), lr=learning_rate, weight_decay=weight_decay)

Deep_Q_Target_Network = SequentialMultiLayerNN(4,128,3,1)
'''Deep_Q_Target_Network：目标网络'''

Deep_Q_Target_Network.load_state_dict(Deep_Q_Network.state_dict())
Deep_Q_Network.to('cuda')
Deep_Q_Target_Network.to('cuda')

path = 'Experience/DataSet_12_27_No6_tensors.pth'
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
all_action_batch = all_action.repeat(mini_batch_size, 1)
all_action_batch = all_action_batch.float()
all_action_batch = all_action_batch.to('cuda')
# print(all_action_batch.shape)

Deep_Q_Network.train()
Deep_Q_Target_Network.eval()

start_time = time.time()

prev_best_avg_loss = float('inf')
count_adjust_learning_rate = 0

Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min',
                                                                      patience=patience, factor=factor, min_lr=1e-12,
                                                                      threshold=1e-5, threshold_mode='rel')

for round in range(rounds):

    if (round+1) % 10 == 0:
        Deep_Q_Target_Network.load_state_dict(Deep_Q_Network.state_dict())

    with torch.no_grad():

        indices = torch.randperm(agent_pos_1_tensor.size(0))[:mini_batch_size]
        agent_pos_1_batch = agent_pos_1_tensor[indices]
        agent_pos_n1_batch = agent_pos_n1_tensor[indices]
        tasks_state_1_batch = tasks_state_1_tensor[indices]
        tasks_state_n1_batch = tasks_state_n1_tensor[indices]
        agent_action_1_batch = agent_action_1_tensor[indices]
        trajectory_reward_batch = trajectory_reward_tensor[indices]

        state_1_action_1 = torch.cat((agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch), dim=1)

        state_n1_batch = torch.cat((agent_pos_n1_batch, tasks_state_n1_batch), dim=1)
        state_n1_batch = state_n1_batch.repeat_interleave(9, dim=0)
        state_n1_all_action_batch = torch.cat((state_n1_batch, all_action_batch), dim=1)

        q_value_state_n1_all_action_batch = Deep_Q_Target_Network(state_n1_all_action_batch)
        reshaped_q_value_batch = q_value_state_n1_all_action_batch.view(-1, 9)
        max_q_values, _ = reshaped_q_value_batch.max(dim=1)
        max_q_s_n1 = max_q_values.view(-1, 1)

        TD_target_batch = trajectory_reward_batch + gamma_power_n * max_q_s_n1

    # print(state_1_action_1)
    TD_error = TD_target_batch - Deep_Q_Network(state_1_action_1)
    # print(TD_error)
    loss = 0.5 * (TD_error ** 2)
    # print(loss)
    loss = loss.mean()
    # print(loss)

    Deep_Q_Network_optimizer.zero_grad()
    loss.backward()
    Deep_Q_Network_optimizer.step()

    '''for param_group in Deep_Q_Network_optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")'''

    several_training_loss.append(loss.item())

    if len(several_training_loss) == several_training_loss.maxlen:
        avg_loss = sum(several_training_loss)/several_training_loss.maxlen
        training_loss_memory.append(avg_loss)
        print("average loss:", avg_loss)
        several_training_loss.clear()
        Deep_Q_Network_scheduler.step(avg_loss)
        print("学习率:",Deep_Q_Network_scheduler.get_last_lr())
        # print("average loss:",avg_loss)

    if (round + 1) % 100 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"百步运行时间: {elapsed_time:.6f} 秒")
        print(f"进度：{100 * (round + 1) / rounds}%\n")
        start_time = time.time()

    '''if len(several_training_loss) == several_training_loss.maxlen:

        several_training_loss.clear()
        
        if prev_best_avg_loss != float('inf'):
            avg_loss_relative_change = (prev_best_avg_loss - avg_loss) / prev_best_avg_loss
        else:
            avg_loss_relative_change = float('inf')  # 第一次循环无相对变化
        print("avg_loss_relative_change:", avg_loss_relative_change)

        if avg_loss < prev_best_avg_loss:
            prev_best_avg_loss = avg_loss

        if avg_loss_relative_change < 1e-4:
            count_adjust_learning_rate += 1
        else: count_adjust_learning_rate = 0

        print("count_adjust_learning_rate:",count_adjust_learning_rate,"\n")

        if count_adjust_learning_rate >= 3:
            learning_rate = learning_rate * 0.5
            if learning_rate < 1e-10:
                learning_rate = 1e-10
            print('\nlearning_rate:', learning_rate,"\n")
            count_adjust_learning_rate = 0
            prev_best_avg_loss = float('inf')'''


path = "model/Deep_Q_network_12_28_No2.pth"
torch.save(Deep_Q_Network.state_dict(), path)

x = range(len(training_loss_memory))
plt.plot(x, training_loss_memory)
plt.title('training_loss')
plt.xlabel('training_times')
plt.ylabel('training_loss')
plt.show()
