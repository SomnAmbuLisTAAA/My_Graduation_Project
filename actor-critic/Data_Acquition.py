import numpy as np
import matplotlib.pyplot as plt
import torch

from Environment_v1 import MultiAgentTrainingEnv_2D_v0

agent_init_pos = np.array([ [5.0,5.0],[25.0,5.0] ])
agent_radius = np.array([ [0.2],[0.2] ])

device = 'cuda'

change_agent_pos_count_1 = 0
change_agent_pos_count_2 = 0
change_agent_pos_count_3 = 0
change_agent_pos_count_4 = 0

init_pos_index = 0
agent_init_pos_list = [np.array([ [25.0,5.0] ]), np.array([ [5.0,5.0] ]), np.array([ [5.0,25.0] ]), np.array([ [25.0,25.0] ]),
                       np.array([ [12.5,22.5] ]), np.array([ [17.5,7.5] ]), np.array([ [25.0,25.0] ])]

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
task_radius = np.array([ [0.5],[0.5] ])

env = MultiAgentTrainingEnv_2D_v0(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,
                                            task_pos,task_radius, accuracy=0.5, multi_step=10)

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_init_pos[0][0])
agent1_y_list.append(agent_init_pos[0][1])

agent2_x_list = []
agent2_y_list = []
agent2_x_list.append(agent_init_pos[1][0])
agent2_y_list.append(agent_init_pos[1][1])

final_size = 1000000

agent_pos_1_matrix = np.empty((final_size, 4))
agent_pos_n1_matrix = np.empty((final_size, 4))
tasks_state_1_matrix = np.empty((final_size, 5))
tasks_state_n1_matrix = np.empty((final_size, 5))
agent_action_1_matrix = np.empty((final_size, 4))
trajectory_reward_matrix = np.empty((final_size, 2))

line = 0

while line != final_size:

    (agent_pos_1_new_row, agent_pos_n1_new_row, tasks_state_1_new_row, tasks_state_n1_new_row, agent_action_1_new_row,
     trajectory_reward_new_row) = env.step()

    if not agent_pos_n1_new_row.size == 0:

        # print(agent_action_1_new_row)

        trajectory_reward_matrix[line] = trajectory_reward_new_row
        agent_action_1_matrix[line] = agent_action_1_new_row
        agent_pos_1_matrix[line] = agent_pos_1_new_row
        agent_pos_n1_matrix[line] = agent_pos_n1_new_row
        tasks_state_1_matrix[line] = tasks_state_1_new_row
        tasks_state_n1_matrix[line] = tasks_state_n1_new_row
        line += 1

        agent1_x_list.append(agent_pos_1_new_row[0])
        agent1_y_list.append(agent_pos_1_new_row[1])
        agent2_x_list.append(agent_pos_1_new_row[2])
        agent2_y_list.append(agent_pos_1_new_row[3])

        # print(tasks_state_1_new_row[4])

        '''print(agent_pos_1_new_row)
        print(agent_action_1_new_row)
        print(tasks_state_1_new_row)
        print(trajectory_reward_new_row)'''

        '''if 0.25 >= (line + 1) / final_size > 0 == tasks_state_1_new_row[0]:

            change_agent_pos_count_1 += 1

            if change_agent_pos_count_1 >= 100000:
                init_pos_index = init_pos_index % 7
                print("new pos:\n", agent_init_pos_list[init_pos_index])
                env.change_agent_pos(agent_init_pos_list[init_pos_index])
                init_pos_index += 1
                change_agent_pos_count_1 = 0'''

        if 0.25 < (line + 1) / final_size <=0.50 and (tasks_state_1_new_row[4] == 0 or tasks_state_1_new_row[4] == 1):

            '''change_agent_pos_count_2 += 1

            if change_agent_pos_count_2 >= 10:
                init_pos_index = init_pos_index % 7
                print("new pos:\n", agent_init_pos_list[init_pos_index])
                env.change_agent_pos(agent_init_pos_list[init_pos_index])
                init_pos_index += 1
                change_agent_pos_count_2 = 0'''
            env.get_reward_times()
            env.reset_task([0])
            print("任务1重置")

        if 0.50 < (line + 1) / final_size <= 0.75 and (tasks_state_1_new_row[4] == 0 or tasks_state_1_new_row[4] == 2):

            '''change_agent_pos_count_3 += 1

            if change_agent_pos_count_3 >= 10:
                init_pos_index = init_pos_index % 7
                print("new pos:\n", agent_init_pos_list[init_pos_index])
                env.change_agent_pos(agent_init_pos_list[init_pos_index])
                init_pos_index += 1
                change_agent_pos_count_3 = 0'''
            env.get_reward_times()
            env.reset_task([1])
            print("任务2重置")

        if 0.75 < (line + 1) / final_size and tasks_state_1_new_row[4] != 3:
            env.get_reward_times()
            env.reset_task()
            print("任务重置")

    if (line+1)%1000==0:
        print(f"进度:{(100 * (line+1)/final_size):.2f}%")

trajectory_reward_tensor = torch.from_numpy(trajectory_reward_matrix)
trajectory_reward_tensor = trajectory_reward_tensor.to(device)

agent_pos_n1_tensor = torch.from_numpy(agent_pos_n1_matrix)
agent_pos_n1_tensor = agent_pos_n1_tensor.to(device)

agent_pos_1_tensor = torch.from_numpy(agent_pos_1_matrix)
agent_pos_1_tensor = agent_pos_1_tensor.to(device)

agent_action_1_tensor = torch.from_numpy(agent_action_1_matrix)
agent_action_1_tensor = agent_action_1_tensor.to(device)

tasks_state_1_tensor = torch.from_numpy(tasks_state_1_matrix)
tasks_state_1_tensor = tasks_state_1_tensor.to(device)

tasks_state_n1_tensor = torch.from_numpy(tasks_state_n1_matrix)
tasks_state_n1_tensor = tasks_state_n1_tensor.to(device)

print(trajectory_reward_matrix)
print(agent_pos_1_matrix)
print(agent_pos_n1_matrix)
print(agent_action_1_matrix)
print(tasks_state_1_matrix)
print(tasks_state_n1_matrix)



path = 'Experience/DataSet_01_10_No1_tensors.pth'
torch.save({'trajectory_reward_tensor': trajectory_reward_tensor, 'agent_pos_n1_tensor': agent_pos_n1_tensor,
            'agent_pos_1_tensor': agent_pos_1_tensor,'agent_action_1_tensor':agent_action_1_tensor,
            'tasks_state_1_tensor': tasks_state_1_tensor, 'tasks_state_n1_tensor': tasks_state_n1_tensor}, path)
print("张量存储完成")



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