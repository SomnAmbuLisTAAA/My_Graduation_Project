from DeepLearning.NN_model import SequentialMultiLayerNN, SequentialMultiLayerNN_with_scaled_tanh
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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

task_pos = np.array([ [5.4,10.3],[2.8,10.2] ])
task_radius = np.array([ [0.5],[0.5] ])

actor_network = SequentialMultiLayerNN_with_scaled_tanh(4,128,3,2,1,0.1)

path = ["DeepLearning/model/12_25/actor_network_05.pth",
        "DeepLearning/model/12_25/central_critic_network_05.pth"]
actor_network.load_state_dict(torch.load(path[0], weights_only=True))

sample_time = 0.05
steps = 1000

agent_pos = agent_init_pos.copy()

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_pos[0][0])
agent1_y_list.append(agent_pos[0][1])

agent2_x_list = []
agent2_y_list = []
agent2_x_list.append(agent_pos[1][0])
agent2_y_list.append(agent_pos[1][1])

unfinished_task = np.ones((1, task_pos.shape[0]), dtype=int)


def update_task_state():

    dis_agent2task = cdist(agent_pos, task_pos)
    task_radius_matrix = np.tile(task_radius.T, (agent_init_pos.shape[0], 1))
    '''print('unfinished_tasks:\n', unfinished_tasks)
    print("agent2task:\n", dis_agent2task)
    print("task_radius_matrix:\n", task_radius_matrix)'''
    if_reach_task_region = dis_agent2task <= task_radius_matrix
    # print("if_reach_task_region:\n", if_reach_task_region)
    if_task_finished = np.any(if_reach_task_region, axis=0)
    # print("if_task_finished:\n", if_task_finished)
    unfinished_tasks = (~ if_task_finished) & unfinished_task
    # print('unfinished_tasks:\n', unfinished_tasks)
    return unfinished_tasks


actor_network.eval()
for step in range(steps):

    with torch.no_grad():
        unfinished_task = update_task_state()
        observation_1 = torch.tensor(np.hstack((agent_pos[0:1, :], unfinished_task)), dtype=torch.float32)
        observation_2 = torch.tensor(np.hstack((agent_pos[1:2, :], unfinished_task)), dtype=torch.float32)
        print('observation_1:', observation_1)
        # print('observation_2:', observation_2)
        agent1_action = actor_network(observation_1).numpy()
        agent2_action = actor_network(observation_2).numpy()
        #print("agent1_action:", agent1_action)
        # print("agent2_action:", agent2_action)
        agent_v = np.vstack((agent1_action, agent2_action))
        # print("agent_v:\n", agent_v)
        agent_pos += agent_v * sample_time

        agent1_x_list.append(agent_pos[0][0])
        agent1_y_list.append(agent_pos[0][1])
        agent2_x_list.append(agent_pos[1][0])
        agent2_y_list.append(agent_pos[1][1])




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