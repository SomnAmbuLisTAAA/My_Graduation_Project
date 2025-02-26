import torch
import numpy as np
import matplotlib.pyplot as plt
import time
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

task_pos = np.array([ [7.5,17.5],[12.5,12.5],[17.5,17.5],[17.5,22.5] ])
task_radius = np.array([ [0.5],[0.5],[0.5],[0.5] ])

env = ContinuousMultiAgentTrainingEnv_2D_v1(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,
                                            task_pos,task_radius,sample_time=1)

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_init_pos[0][0])
agent1_y_list.append(agent_init_pos[0][1])

agent2_x_list = []
agent2_y_list = []
agent2_x_list.append(agent_init_pos[1][0])
agent2_y_list.append(agent_init_pos[1][1])

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

while line != final_size:

    if env.if_done():
        # print(env.return_unfinished_tasks())
        env.reset_random_pos()
        # print(env.return_unfinished_tasks())

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

    agent1_x_list.append(pos[0][0])
    agent1_y_list.append(pos[0][1])
    agent2_x_list.append(pos[1][0])
    agent2_y_list.append(pos[1][1])
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
        print(f"进度: [{100 * (line + 1) / final_size}%]")
        start_time = time.time()

'''end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.6f} 秒")'''

# env.get_reward_times()

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

'''print('trajectory_reward_tensor:',trajectory_reward_tensor)
print('sn1_tensor',sn1_tensor)
print('s1_a1_tensor',s1_a1_tensor)'''

print(trajectory_reward_tensor.shape)
print(agent_pos_n1_tensor.shape)
print(agent_pos_1_tensor.shape)
print(agent_action_1_tensor.shape)
print(tasks_state_1_tensor.shape)
print(tasks_state_n1_tensor.shape)

print(trajectory_reward_tensor)
print(agent_pos_n1_tensor)
print(agent_pos_1_tensor)
print(agent_action_1_tensor)
print(tasks_state_1_tensor)
print(tasks_state_n1_tensor)

# 创建一个图形和子图
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



'''
start_time = time.time()
rounds = 1000000

# 示例使用
agent1 = agent.DQN.Agent(2,2,0.5,-10.0,10.0,-10.0,10.0,5.0,10.0)
agent2 = agent.DQN.Agent(2,2,0.5,-10.0,10.0,-10.0,10.0,25.0,20.0)
agent_group = [agent1,agent2]
# obstacle_positions = [(10.0, 22.0), (23.0, 8.0), (25.0, 25.0), (7.0,7.0)]  # 障碍物位置
obstacle_positions = np.array([ [5.0,5.0], [5.0,15.0], [5.0,25.0],
                     [10.0,5.0],[10.0,15.0],[10.0,25.0],
                     [15.0,5.0],[15.0,15.0],[15.0,25.0],
                     [20.0,5.0],[20.0,15.0],[20.0,25.0],
                     [25.0,5.0],[25.0,15.0],[25.0,25.0] ])
goal_position = [(10.0, 10.0), (20.0, 20.0)]  # 目标点位置
env = ContinuousEnv(agent_group,1.0,30.0,30.0,obstacle_positions,goal_position,sample_time=0.05)
env.reset()
env.render()

agent1_pos_list = []
agent2_pos_list = []

agent_pos = env.agent_group[0].get_agent_pos()
agent1_pos_list.append(tuple(agent_pos.tolist()))
agent_pos = env.agent_group[1].get_agent_pos()
agent2_pos_list.append(tuple(agent_pos.tolist()))


for round in range(rounds):

    env.upgrade_pos()
    # env.upgrade_pos_behavior_policy()
    env.get_reward()
    agent_pos = env.agent_group[0].get_agent_pos()
    agent1_pos_list.append(tuple(agent_pos.tolist()))
    agent_pos = env.agent_group[1].get_agent_pos()
    agent2_pos_list.append(tuple(agent_pos.tolist()))
    if (round + 1) % 1000 == 0:
        print(f"Rounds: [{100 * (round + 1) / rounds}%]")

    # env.return_agent_position()

#print(agent1_pos_list)
#print(agent2_pos_list)

env.print_negative_reward_time()
env.print_reward()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.6f} 秒")

# 解包坐标点列表（拆分为x和y）
line1_x, line1_y = zip(*agent1_pos_list)  # 将line1的x和y坐标分开
line2_x, line2_y = zip(*agent2_pos_list)  # 将line2的x和y坐标分开

# 创建一个图形和子图
fig, ax = plt.subplots()

# 绘制散点图
ax.scatter(line1_x, line1_y, marker='o', s=0.1, label="Agent 1")  # 第一组散点

# 如果需要，可以添加第二组数据的散点
ax.scatter(line2_x, line2_y, marker='o', s=0.1, label="Agent 2")

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
'''