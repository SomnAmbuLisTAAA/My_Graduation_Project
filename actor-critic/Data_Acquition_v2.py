import numpy as np
import torch
import matplotlib.pyplot as plt

from Environment_v3 import MultiAgentTrainingEnv_2D_v3

agent_init_pos = np.array([ [5.0,5.0],[25.0,5.0] ])
agent_radius = np.array([ [0.1],[0.1] ])

device = 'cuda'

init_pos_index = 0
agent_init_pos_list = [np.array([ [25.0,5.0] ]), np.array([ [5.0,5.0] ]), np.array([ [5.0,25.0] ]), np.array([ [25.0,25.0] ]),
                       np.array([ [12.5,22.5] ]), np.array([ [17.5,7.5] ]), np.array([ [25.0,25.0] ])]

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


env = MultiAgentTrainingEnv_2D_v3(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,
                                            task_pos,task_radius, accuracy=0.25, multi_step=5)

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_init_pos[0][0])
agent1_y_list.append(agent_init_pos[0][1])

agent2_x_list = []
agent2_y_list = []
agent2_x_list.append(agent_init_pos[1][0])
agent2_y_list.append(agent_init_pos[1][1])

final_size = 8000000
task_state_num = 16

task_reset_scalar = final_size / task_state_num

trajectory_reward_matrix = np.empty((final_size, 2), dtype=np.float32)
o1_1_matrix = np.empty((final_size, 7) ,dtype=np.float32)
o2_1_matrix = np.empty((final_size, 7) ,dtype=np.float32)
o1_n1_matrix = np.empty((final_size, 7) ,dtype=np.float32)
o2_n1_matrix = np.empty((final_size, 7) ,dtype=np.float32)
state_1_matrix = np.empty((final_size, 9) ,dtype=np.float32)
state_1_without_actions_matrix = np.empty((final_size, 5) ,dtype=np.float32)
state_n1_without_actions_matrix = np.empty((final_size, 5) ,dtype=np.float32)

line = 0

while line != final_size:

    observation_1 ,observation_n1, state_1, state_1_without_actions, state_n1_without_actions, trajectory_reward = env.step()

    if not observation_1.size == 0:

        # print(agent_action_1_new_row)

        trajectory_reward_matrix[line] = trajectory_reward
        o1_1_matrix[line] = observation_1[0]
        o2_1_matrix[line] = observation_1[1]
        o1_n1_matrix[line] = observation_n1[0]
        o2_n1_matrix[line] = observation_n1[1]
        state_1_matrix[line] = state_1
        state_1_without_actions_matrix[line] = state_1_without_actions
        state_n1_without_actions_matrix[line] = state_n1_without_actions
        line += 1

        agent1_x_list.append(observation_1[0][0])
        agent1_y_list.append(observation_1[0][1])
        agent2_x_list.append(observation_1[1][0])
        agent2_y_list.append(observation_1[1][1])

        '''if observation_1[0][4] == 0:
            env.get_reward_times()
            env.reset_task()
            print("任务重置")'''

        '''print(trajectory_reward)
        print(observation_1)
        print(observation_n1)'''

        '''print(agent_pos_1_new_row)
        print(agent_action_1_new_row)
        print(tasks_state_1_new_row)
        print(trajectory_reward_new_row)'''

        ''''if 0.25 < (line + 1) / final_size <=0.50 and (observation_1[0][4] == 0 or observation_1[0][4] == 1):
            env.get_reward_times()
            env.reset_task([0])
            print("任务1重置")

        if 0.50 < (line + 1) / final_size <= 0.75 and (observation_1[0][4] == 0 or observation_1[0][4] == 2):
            env.get_reward_times()
            env.reset_task([1])
            print("任务2重置")'''

        if observation_1[0][6] != 1 and task_reset_scalar < (line + 1) <= task_reset_scalar * 2:
            env.get_reward_times()
            env.reset_task([3])
            print("任务4重置")

        if observation_1[0][6] != 2 and task_reset_scalar * 2 < (line + 1) <= task_reset_scalar * 3:
            env.get_reward_times()
            env.reset_task([2])
            print("任务3重置")

        if observation_1[0][6] != 3 and task_reset_scalar * 3 < (line + 1) <= task_reset_scalar * 4:
            env.get_reward_times()
            env.reset_task([2,3])
            print("任务3,4重置")

        if observation_1[0][6] != 4 and task_reset_scalar * 4 < (line + 1) <= task_reset_scalar * 5:
            env.get_reward_times()
            env.reset_task([1])
            print("任务2重置")

        if observation_1[0][6] != 5 and task_reset_scalar * 5 < (line + 1) <= task_reset_scalar * 6:
            env.get_reward_times()
            env.reset_task([1,3])
            print("任务2,4重置")

        if observation_1[0][6] != 6 and task_reset_scalar * 6 < (line + 1) <= task_reset_scalar * 7:
            env.get_reward_times()
            env.reset_task([1,2])
            print("任务2,3重置")

        if observation_1[0][6] != 7 and task_reset_scalar * 7 < (line + 1) <= task_reset_scalar * 8:
            env.get_reward_times()
            env.reset_task([1,2,3])
            print("任务2,3,4重置")

        if observation_1[0][6] != 8 and task_reset_scalar * 8 < (line + 1) <= task_reset_scalar * 9:
            env.get_reward_times()
            env.reset_task([0])
            print("任务1重置")

        if observation_1[0][6] != 9 and task_reset_scalar * 9 < (line + 1) <= task_reset_scalar * 10:
            env.get_reward_times()
            env.reset_task([0,3])
            print("任务1,4重置")

        if observation_1[0][6] != 10 and task_reset_scalar * 10 < (line + 1) <= task_reset_scalar * 11:
            env.get_reward_times()
            env.reset_task([0,2])
            print("任务1,3重置")

        if observation_1[0][6] != 11 and task_reset_scalar * 11 < (line + 1) <= task_reset_scalar * 12:
            env.get_reward_times()
            env.reset_task([0,2,3])
            print("任务1,3,4重置")

        if observation_1[0][6] != 12 and task_reset_scalar * 12 < (line + 1) <= task_reset_scalar * 13:
            env.get_reward_times()
            env.reset_task([0,1])
            print("任务1,2重置")

        if observation_1[0][6] != 13 and task_reset_scalar * 13 < (line + 1) <= task_reset_scalar * 14:
            env.get_reward_times()
            env.reset_task([0,1,3])
            print("任务1,2,4重置")

        if observation_1[0][6] != 14 and task_reset_scalar * 14 < (line + 1) <= task_reset_scalar * 15:
            env.get_reward_times()
            env.reset_task([0,1,2])
            print("任务1,2,3重置")

        if observation_1[0][6] != 15 and task_reset_scalar * 15 < (line + 1):
            env.get_reward_times()
            env.reset_task()
            print("任务重置")

    if (line+1)%1000==0:
        print(f"进度:{(100 * (line+1)/final_size):.2f}%")

trajectory_reward_tensor = torch.from_numpy(trajectory_reward_matrix)
o1_1_tensor = torch.from_numpy(o1_1_matrix)
o2_1_tensor = torch.from_numpy(o2_1_matrix)
o1_n1_tensor = torch.from_numpy(o1_n1_matrix)
o2_n1_tensor = torch.from_numpy(o2_n1_matrix)
state_1_tensor = torch.from_numpy(state_1_matrix)
state_1_without_actions_tensor = torch.from_numpy(state_1_without_actions_matrix)
state_n1_without_actions_tensor = torch.from_numpy(state_n1_without_actions_matrix)

trajectory_reward_tensor = trajectory_reward_tensor.to(device)
o1_1_tensor = o1_1_tensor.to(device)
o2_1_tensor = o2_1_tensor.to(device)
o1_n1_tensor = o1_n1_tensor.to(device)
o2_n1_tensor = o2_n1_tensor.to(device)
state_1_tensor = state_1_tensor.to(device)
state_1_without_actions_tensor = state_1_without_actions_tensor.to(device)
state_n1_without_actions_tensor = state_n1_without_actions_tensor.to(device)

path = 'Experience/DataSet_02_22_No1_tensors.pth'
torch.save({'trajectory_reward_tensor': trajectory_reward_tensor, 'o1_1_tensor': o1_1_tensor,
            'o2_1_tensor': o2_1_tensor, 'o1_n1_tensor':o1_n1_tensor, 'o2_n1_tensor': o2_n1_tensor,
            'state_1_tensor': state_1_tensor,'state_1_without_actions_tensor': state_1_without_actions_tensor,
            'state_n1_without_actions_tensor': state_n1_without_actions_tensor}, path)
print("张量存储完成")

print(trajectory_reward_tensor)
print(o1_1_tensor)
print(o2_1_tensor)
print(o1_n1_tensor)
print(o2_n1_tensor)
print(state_1_tensor)
print(state_1_without_actions_tensor)
print(state_n1_without_actions_tensor)


fig, ax = plt.subplots()
# 绘图
# ax.plot(agent1_x_list, agent1_y_list, marker='o', markersize=1, label="Agent 1")
ax.scatter(agent1_x_list, agent1_y_list, marker='o', s=0.1, label="Agent 1")
ax.scatter(agent2_x_list, agent2_y_list, marker='o', s=0.1, label="Agent 2")
# 设置坐标轴范围
ax.set_xlim(0.0, 30.0)  # 设置x轴的范围为0到30
ax.set_ylim(0.0, 30.0)  # 设置y轴的范围为0到30
# 设置xy轴等比例
ax.set_aspect('equal')
# 添加图例和标题
ax.legend(loc='upper left')
ax.set_title("Scatter Plot of Agent Positions")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
# 显示网格
ax.grid(True)
# 显示图形
plt.show()