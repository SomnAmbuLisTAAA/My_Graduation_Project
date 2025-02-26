import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from Neural_network_model import SequentialMultiLayerNN

agent_pos = np.array([ [5.0,5.0],[25.0,5.0] ])

task_pos = np.array([ [7.5,17.5],[17.5,17.5] ])
task_pos_tensor = torch.tensor(task_pos.flatten(), dtype=torch.float32).reshape(1,-1)
# print(task_pos_tensor)

task_radius = np.array([ [0.8,0.8] ])
task_radius_matrix = np.tile(task_radius, (agent_pos.shape[0], 1))
print(task_radius_matrix)

unfinished_task = np.array( [[1],[1]], dtype=np.int32)

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

done = False
accuracy = 0.5
step = 0

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_pos[0][0])
agent1_y_list.append(agent_pos[0][1])

agent2_x_list = []
agent2_y_list = []
agent2_x_list.append(agent_pos[1][0])
agent2_y_list.append(agent_pos[1][1])

Actor_Network = SequentialMultiLayerNN(11,128,3,2,0.6,2.0)
Actor_Network.eval()

path = "Model/actor_01_11_No2.pth"
Actor_Network.load_state_dict(torch.load(path, weights_only=True))

while not done:

    with torch.no_grad():

        agent1_pos_tensor = torch.from_numpy(agent_pos[0]).reshape(1, -1).float()
        agent2_pos_tensor = torch.from_numpy(agent_pos[1]).reshape(1, -1).float()

        dis_agent2task = cdist(agent_pos, task_pos)
        # print(dis_agent2task)

        if_reach_task_region = dis_agent2task <= task_radius_matrix
        # print(if_reach_task_region)
        if_reach_task_region = np.any(if_reach_task_region, axis=0).reshape(-1, 1)
        # print(if_reach_task_region)

        unfinished_task = (~ if_reach_task_region) & unfinished_task
        # print(unfinished_task)

        task_state_decimal = np.dot(unfinished_task.T, 2 ** np.arange(unfinished_task.size)[::-1])[0]
        print(task_state_decimal)

        # o1_tensor = torch.cat((agent1_pos_tensor, task_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        # o2_tensor = torch.cat((agent2_pos_tensor, task_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        o1_tensor = torch.cat((agent1_pos_tensor, agent1_pos_tensor, agent2_pos_tensor, task_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        o2_tensor = torch.cat((agent2_pos_tensor, agent1_pos_tensor, agent2_pos_tensor, task_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        # o1_tensor = torch.cat((agent1_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        # o2_tensor = torch.cat((agent2_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        print(o1_tensor)

        u_o1_tensor = Actor_Network.with_scaled_tanh(o1_tensor)
        u_o2_tensor = Actor_Network.with_scaled_tanh(o2_tensor)

        agent_v_tensor = torch.cat((u_o1_tensor, u_o2_tensor), dim=0)
        agent_v = agent_v_tensor.numpy()
        print(agent_v)

        agent_pos += agent_v * accuracy

        agent1_x_list.append(agent_pos[0][0])
        agent1_y_list.append(agent_pos[0][1])
        agent2_x_list.append(agent_pos[1][0])
        agent2_y_list.append(agent_pos[1][1])

        step += 1
        if task_state_decimal == 0: done = True
        if step >= 1000: done = True

        if done:
            break


print(step)
fig, ax = plt.subplots()
# 绘图
ax.plot(agent1_x_list, agent1_y_list, marker='o', markersize=1, label="Agent 1")
ax.plot(agent2_x_list, agent2_y_list, marker='o', markersize=1, label="Agent 2")
# ax.scatter(agent1_x_list, agent1_y_list, marker='o', s=1, label="Agent 1")

for center, radius in zip(obs_pos, obs_radius):
    circle = plt.Circle(center, radius, color='red', fill=True)
    ax.add_artist(circle)

for center1, radius1 in zip(task_pos, task_radius.T):
    circle1 = plt.Circle(center1, radius1, color='yellow', fill=True)
    ax.add_artist(circle1)

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