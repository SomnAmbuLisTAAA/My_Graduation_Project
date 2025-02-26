import torch
import numpy as np
import time

from gmpy2.gmpy2 import gamma

from ReinforcementLearning.TestEnvironment import ContinuousMultiAgentTrainingEnv_2D_v1

agent_init_pos = np.array([ [5.0,5.0],[25.0,5.0] ])
agent_radius = np.array([ [0.2],[0.2] ])

obs_pos = np.array([                        [5.0,15.0],
                                [10.0,10.0],[10.0,15.0],[10.0,20.0],
                     [15.0,5.0],[15.0,10.0],[15.0,15.0],[15.0,20.0],[15.0,25.0],
                                [20.0,10.0],[20.0,15.0],[20.0,20.0],
                                            [25.0,15.0]
                    ])

obs_radius = np.array([       [1.0],
                        [1.0],[1.0],[1.0],
                  [1.0],[1.0],[1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],
                              [1.0]
                    ])

task_pos = np.array([ [12.5,12.5],[17.5,17.5] ])
task_radius = np.array([ [0.5],[0.5] ])

env = ContinuousMultiAgentTrainingEnv_2D_v1(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,
                                            task_pos,task_radius,sample_time=1.0,multi_step=10,gamma=0.99)

final_size = 2**20

trajectory_reward_matrix = np.empty((final_size, 1))
tasks_state_n1_matrix = np.empty((final_size, task_pos.shape[0]), dtype=int)
tasks_state_1_matrix = np.empty((final_size, task_pos.shape[0]), dtype=int)
agent_pos_n1_matrix = np.empty((final_size, 4))
agent_pos_1_matrix = np.empty((final_size, 4))
agent_action_1_matrix = np.empty((final_size, 4))

# print(task_pos.shape[0])

round = 1
line = 0
start_time = time.time()

# env.reset_random_pos()
while line != final_size:

    if env.if_done():
        env.reset_random_pos()

    # env.parameter_check_all()
    v = env.explore_action(-1.0, 1.0, -1.0, 1.0)
    # print("v:",v.flatten())
    env.update_pos_by_action()
    # print("position:",pos)
    immediate_reward, accumulate_reward, pos = env.calculate_reward()
    (agent_pos_1_new_row, agent_action_1_new_row, agent_pos_n1_new_row, trajectory_reward_new_row,
     tasks_state_1_new_row, tasks_state_n1_new_row) = env.get_experience()
    if not agent_pos_n1_new_row.size == 0:
        trajectory_reward_matrix[line] = trajectory_reward_new_row
        agent_action_1_matrix[line] = agent_action_1_new_row
        agent_pos_1_matrix[line] = agent_pos_1_new_row
        agent_pos_n1_matrix[line] = agent_pos_n1_new_row
        tasks_state_1_matrix[line] = tasks_state_1_new_row
        tasks_state_n1_matrix[line] = tasks_state_n1_new_row
        line += 1
        '''print(tasks_state_1_new_row)
        print(tasks_state_n1_new_row)'''

    # print("accumulate_reward:\n", np.sum(accumulate_reward).item())

    '''print('trajectory_reward_matrix:',trajectory_reward_matrix)
    print('sn1_matrix',sn1_matrix)
    print('s1_a1_matrix:',s1_a1_matrix)'''

    # print(agent1_x_list)
    # print("immediate_reward:",np.sum(immediate_reward).item())
    # print("position:",pos.flatten())
    '''print("s1_a1:",s1_a1)
    print("sn1:", sn1)
    print("trajectory_reward:", trajectory_reward)'''
    # env.parameter_check_dis()
    # env.parameter_check_reward()
    round += 1
    if round % 10000 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"万步运行时间: {elapsed_time:.6f} 秒")
        print(f"进度: [{(100 * (line + 1) / final_size):.2f}%]")
        start_time = time.time()


# env.get_reward_times()
print("数据采集完成")

trajectory_reward_tensor = torch.from_numpy(trajectory_reward_matrix)
trajectory_reward_tensor = trajectory_reward_tensor.to('cuda')

agent_pos_n1_tensor = torch.from_numpy(agent_pos_n1_matrix)
agent_pos_n1_tensor = agent_pos_n1_tensor.to('cuda')

agent_pos_1_tensor = torch.from_numpy(agent_pos_1_matrix)
agent_pos_1_tensor = agent_pos_1_tensor.to('cuda')

agent_action_1_tensor = torch.from_numpy(agent_action_1_matrix)
agent_action_1_tensor = agent_action_1_tensor.to('cuda')

tasks_state_1_tensor = torch.from_numpy(tasks_state_1_matrix)
tasks_state_1_tensor = tasks_state_1_tensor.to('cuda')

tasks_state_n1_tensor = torch.from_numpy(tasks_state_n1_matrix)
tasks_state_n1_tensor = tasks_state_n1_tensor.to('cuda')

print("张量转化完成")

'''print('trajectory_reward_tensor:',trajectory_reward_tensor)
print('sn1_tensor',sn1_tensor)
print('s1_a1_tensor',s1_a1_tensor)'''

print(trajectory_reward_tensor.shape)
print(agent_pos_n1_tensor.shape)
print(agent_pos_1_tensor.shape)
print(agent_action_1_tensor.shape)
print(tasks_state_1_tensor.shape)
print(tasks_state_n1_tensor.shape)

path = 'ReinforcementLearning/Experience/DataSet_12_25_No3_tensors.pth'
torch.save({'trajectory_reward_tensor': trajectory_reward_tensor, 'agent_pos_n1_tensor': agent_pos_n1_tensor,
            'agent_pos_1_tensor': agent_pos_1_tensor,'agent_action_1_tensor':agent_action_1_tensor,
            'tasks_state_1_tensor': tasks_state_1_tensor, 'tasks_state_n1_tensor': tasks_state_n1_tensor}, path)
print("张量存储完成")

'''
buffer = PrioritizedReplayBuffer(rounds-9,4,4,10,0.6,0.4,0.95,1e-6)
buffer.load_from_file(path)
priorities = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], device='cuda')
buffer.set_priorities(priorities)
buffer.print_buffer_info()
buffer.print_buffer_data()

s1_a1_batch, sn1_batch, trajectory_reward_batch, weight_batch = buffer.sample(16)
print('trajectory_reward_batch:\n', trajectory_reward_batch)
print('sn1_batch:\n', sn1_batch)
print('s1_a1_batch:\n', s1_a1_batch)
print('weight_batch:\n', weight_batch)
'''

'''
times = []

for _ in range(100):
    s1_a1_batch, sn1_batch, trajectory_reward_batch, weight_batch = buffer.sample(16)


    # 检查 sn1_batch 中的每一行是否在 sn1_tensor 中出现
    matching_indices = []
    time = 0

    for i, batch_row in enumerate(sn1_batch):
        for j, tensor_row in enumerate(sn1_tensor):
            if torch.allclose(batch_row, tensor_row, atol=1e-4):  # 使用一个小的公差来检查是否匹配
                matching_indices.append((i, j))
                break

    # 输出匹配的行
    # print("匹配结果：")
    for i, j in matching_indices:
        # print(f"sn1_batch 的第 {i} 行 与 sn1_tensor 的第 {j} 行 相同")
        if j > 7:
            time += 1

    # print('time:', time)
    times.append(time)

count = len([x for x in times if x > 8])
print(count)
'''