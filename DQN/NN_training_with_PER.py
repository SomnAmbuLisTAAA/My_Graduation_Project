import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import gc

from Experience_Replay import PrioritizedReplayBuffer
from LR_Scheduler import RecentLossLRScheduler
from NN_model import SequentialMultiLayerNN

mini_batch_size = 1024
init_lr = 0.1
basic_lr = init_lr
current_lr = init_lr

weight_decay = 0.001                      # L2 正则化强度（权重衰减因子）

gamma_power_n = 0.99**10
rounds = 1000

iteration_times = 1

PrioritizedReplay = True
LR_Adjusted = 0

training_loss_memory = []
several_training_loss = deque(maxlen=10)

Deep_Q_Network = SequentialMultiLayerNN(4,128,3,1)
'''Deep_Q_Network：主网络'''
Deep_Q_Network_optimizer = optim.AdamW(Deep_Q_Network.parameters(), lr=init_lr, weight_decay=weight_decay)

Deep_Q_Target_Network_cpu = SequentialMultiLayerNN(4,128,3,1)
'''Deep_Q_Target_Network_cpu：目标网络'''

Deep_Q_Target_Network_cpu.load_state_dict(Deep_Q_Network.state_dict())
Deep_Q_Network.to('cuda')
Deep_Q_Target_Network_cpu.to('cpu')

# basic_lr_adjust = RecentLossLRScheduler(init_lr,0.5,3,10,1e-3,min_lr=1e-7)

buffer = PrioritizedReplayBuffer(2**20,2,1,1,1,10,0.6,0.4,0.99,device='cuda')
path = 'Experience/DataSet_12_31_No2_tensors.pth'
buffer.load_from_file(path)

loaded_tensor_cpu = torch.load(path, weights_only=True, map_location='cpu')
agent_pos_1_tensor_cpu = loaded_tensor_cpu['agent_pos_1_tensor']
agent_pos_n1_tensor_cpu = loaded_tensor_cpu['agent_pos_n1_tensor']
tasks_state_1_tensor_cpu = loaded_tensor_cpu['tasks_state_1_tensor']
tasks_state_n1_tensor_cpu = loaded_tensor_cpu['tasks_state_n1_tensor']
agent_action_1_tensor_cpu = loaded_tensor_cpu['agent_action_1_tensor']

agent_pos_1_tensor_cpu = agent_pos_1_tensor_cpu.float()
agent_pos_n1_tensor_cpu = agent_pos_n1_tensor_cpu.float()
tasks_state_1_tensor_cpu = tasks_state_1_tensor_cpu.float()
tasks_state_n1_tensor_cpu = tasks_state_n1_tensor_cpu.float()
agent_action_1_tensor_cpu = agent_action_1_tensor_cpu.float()

all_action = torch.arange(1, 10).view(-1, 1)
all_action_batch = all_action.repeat(mini_batch_size, 1).float()
all_action_tensor_cpu = all_action.repeat(agent_pos_1_tensor_cpu.size(0), 1).float()
all_action_batch = all_action_batch.to('cuda')

state_1_action_1_tensor_cpu = torch.cat((agent_pos_1_tensor_cpu, tasks_state_1_tensor_cpu, agent_action_1_tensor_cpu), dim=1)
state_n1_tensor_cpu = torch.cat((agent_pos_n1_tensor_cpu, tasks_state_n1_tensor_cpu), dim=1)
state_n1_tensor_cpu = state_n1_tensor_cpu.repeat_interleave(9, dim=0)
state_n1_all_action_tensor_cpu = torch.cat((state_n1_tensor_cpu, all_action_tensor_cpu), dim=1)

del (loaded_tensor_cpu, agent_pos_1_tensor_cpu, agent_pos_n1_tensor_cpu, tasks_state_1_tensor_cpu,
     tasks_state_n1_tensor_cpu, agent_action_1_tensor_cpu, all_action, all_action_tensor_cpu, state_n1_tensor_cpu)
gc.collect()

Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min',
                                                                      patience=10, threshold=1e-3,
                                                                      factor=0.8,min_lr=1e-6)

def update_buffer(PER_mode=True):
    with torch.no_grad():
        Deep_Q_Target_Network_cpu.load_state_dict(Deep_Q_Network.state_dict())

        q_value_state_n1_all_action_tensor_cpu = Deep_Q_Target_Network_cpu(state_n1_all_action_tensor_cpu)
        reshaped_q_value_state_1_tensor_cpu = q_value_state_n1_all_action_tensor_cpu.view(-1, 9)
        max_q_values_state_n1_tensor_cpu, _ = reshaped_q_value_state_1_tensor_cpu.max(dim=1)
        max_q_values_state_n1_tensor_cpu = max_q_values_state_n1_tensor_cpu.view(-1, 1)

        q_value_state_1_action_1_tensor_cpu = Deep_Q_Target_Network_cpu(state_1_action_1_tensor_cpu)

        if PER_mode:
            buffer.update_priorities(max_q_values_state_n1_tensor_cpu.to('cuda'),
                                     q_value_state_1_action_1_tensor_cpu.to('cuda'))
        else:
            buffer.update_TD_info(max_q_values_state_n1_tensor_cpu.to('cuda'),
                                     q_value_state_1_action_1_tensor_cpu.to('cuda'))

update_buffer(PrioritizedReplay)

Deep_Q_Network.train()
Deep_Q_Target_Network_cpu.eval()

start_time = time.time()
for round in range(rounds):

    '''if (round + 1)%150==0:
        Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min',
                                                                              patience=2, threshold=1e-3,
                                                                              factor=0.1, min_lr=1e-5)'''


    with torch.no_grad():

        #start_time2 = time.time()
        if PrioritizedReplay:
            agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch, TD_target_batch, weight_batch = buffer.sample(mini_batch_size)
        else:
            agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch, TD_target_batch, weight_batch = buffer.sample_uniform(mini_batch_size)
        '''end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"采样用时: {elapsed_time2:.6f} 秒")'''

        '''print(agent_pos_1_batch)
        print(tasks_state_1_batch)
        print(agent_action_1_batch)
        print(TD_target_batch)
        print(weight_batch)'''

        state_1_action_1_batch = torch.cat((agent_pos_1_batch, tasks_state_1_batch, agent_action_1_batch), dim=1)

    # Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min',patience=2, threshold=1e-3, factor=0.5)

    for iteration_time in range(iteration_times):

        q_value_state_1_action_1_batch = Deep_Q_Network(state_1_action_1_batch)
        TD_error = TD_target_batch - q_value_state_1_action_1_batch

        if weight_batch is not None:
            loss = ( 0.5 * (TD_error ** 2) * weight_batch ).mean()
        else:
            loss = ( 0.5 * (TD_error ** 2) ).mean()

        several_training_loss.append(loss.item())

        Deep_Q_Network_optimizer.zero_grad()
        loss.backward()
        Deep_Q_Network_optimizer.step()

        print("one step loss:",loss.item())

        # print("实时学习率:",Deep_Q_Network_scheduler.get_last_lr())

        '''for param_group in Deep_Q_Network_optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")'''

    if len(several_training_loss) == several_training_loss.maxlen:

        avg_loss = sum(several_training_loss)/several_training_loss.maxlen
        print("training average loss:", avg_loss)
        several_training_loss.clear()
        # Deep_Q_Network_scheduler.step(avg_loss)
        # print("学习率:", Deep_Q_Network_scheduler.get_last_lr())
        # basic_lr = basic_lr_adjust.step(avg_loss)
        '''for param_group in Deep_Q_Network_optimizer.param_groups:
            param_group['lr'] = basic_lr
        '''
        # basic_lr_adjust.clear_memory()
        # print("基准学习率:", basic_lr)

        total_loss_mean1 = buffer.return_loss_mean().item()
        print("训练前总平均损失:",total_loss_mean1)

        start_time1 = time.time()
        # buffer.adjust_alpha_beta_sigmoid(round+1,rounds,0.6,0.2,0.8,0.4, k=20, p_m=0.5)
        update_buffer(PrioritizedReplay)
        end_time1 = time.time()
        elapsed_time1 = end_time1 - start_time1
        print(f"buffer更新用时: {elapsed_time1:.6f} 秒")

        total_loss_mean2 = buffer.return_loss_mean().item()
        print("训练后总平均损失:", total_loss_mean2)
        delta_total_loss = total_loss_mean2-total_loss_mean1
        training_loss_memory.append(total_loss_mean2)
        print("delta total loss:",delta_total_loss)

        Deep_Q_Network_scheduler.step(total_loss_mean2)
        print("学习率:", Deep_Q_Network_scheduler.get_last_lr())

        if total_loss_mean2 <= 4000:
            break

        '''if LR_Adjusted == 0 and total_loss_mean2 <= 4200:
            LR_Adjusted = 1
            for param_group in Deep_Q_Network_optimizer.param_groups:
                param_group['lr'] *= 0.1
                current_lr = param_group['lr']
            print("学习率1阶段调整完成")
            several_training_loss = deque(maxlen=20)
            Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] = 1e-4
            mini_batch_size = 10000'''

        '''if LR_Adjusted == 1 and total_loss_mean2 <= 3000:
            LR_Adjusted = 2
            for param_group in Deep_Q_Network_optimizer.param_groups:
                param_group['lr'] = 0.001
                current_lr = param_group['lr']
            print("学习率阶段调整完成\n")'''

        # print(avg_loss < total_loss_mean2)
        # if total_loss_mean2 >= total_loss_mean1 or avg_loss < total_loss_mean2:
        '''if (avg_loss < total_loss_mean2 and delta_total_loss >= 10000) or (LR_Adjusted == 1 and delta_total_loss >=0):
            for param_group in Deep_Q_Network_optimizer.param_groups:
                param_group['lr'] *= 0.8
                current_lr = param_group['lr']
                if current_lr <= 1e-6:
                    param_group['lr'] = 1e-6
                    current_lr = param_group['lr']
            Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] *= 0.9'''

       # print("学习率:", current_lr)


        '''if total_loss_mean <= 3600 and LR_Adjusted == False:
            LR_Adjusted = True
            for param_group in Deep_Q_Network_optimizer.param_groups:
                param_group['lr'] = 1e-4
            Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] = 1e-4
            print("学习率与正则化强度调整完成")'''

        '''training_loss_memory.append(total_loss_mean)
        print("学习率:", Deep_Q_Network_scheduler.get_last_lr())'''

        '''
        if PrioritizedReplay == True and Deep_Q_Network_scheduler.get_last_lr()[0] < 1e-3:

            PrioritizedReplay = False
            print("开始进行均匀抽样\n")
            Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] = 1e-3
            print("正则化强度", Deep_Q_Network_optimizer.param_groups[0]['weight_decay'])

            for param_group in Deep_Q_Network_optimizer.param_groups:
                param_group['lr'] = 1e-3
                print(f"Current Learning Rate: {param_group['lr']}")

            Deep_Q_Network_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Deep_Q_Network_optimizer, mode='min',
                                                                                  patience=2, threshold=1e-3,
                                                                                  factor=0.5, min_lr=1e-6)
        '''

    if (round + 1) % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"进度：{100 * (round + 1) / rounds}%")
        print(f"十步运行时间: {elapsed_time:.6f} 秒")
        start_time = time.time()
        # Deep_Q_Network_optimizer.param_groups[0]['weight_decay'] *= 0.95  # 动态调整正则化强度

path = "model/Deep_Q_network_12_31_No2.pth"
torch.save(Deep_Q_Network.state_dict(), path)

x = range(len(training_loss_memory))
plt.plot(x, training_loss_memory)
plt.title('training_loss')
plt.xlabel('training_times')
plt.ylabel('training_loss')
plt.show()

