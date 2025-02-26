import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from NN_model import SequentialMultiLayerNN

agent_pos = np.array([ [5.0,5.0] ], dtype=np.float32)

task_pos = np.array([ [12.5,12.5],[7.5,17.5] ])
task_radius = np.array([ [0.5],[0.5] ])
task_radius = task_radius.T
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
accuracy = 0.1
step = 0

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_pos[0][0])
agent1_y_list.append(agent_pos[0][1])

Q_Network = SequentialMultiLayerNN(4,256,4,1)
Q_Network.eval()

path = "model/Deep_Q_network_01_04_No2.pth"
Q_Network.load_state_dict(torch.load(path, weights_only=True))

all_action = torch.arange(1, 10).view(-1, 1)

while not done:

    with torch.no_grad():

        agent_pos_tensor = torch.from_numpy(agent_pos)

        dis_agent2task = cdist(agent_pos, task_pos)
        # print(dis_agent2task)
        if_reach_task_region = dis_agent2task <= task_radius
        if_reach_task_region = if_reach_task_region.reshape(-1,1)
        # print(if_reach_task_region)
        # print(if_reach_task_region.shape)
        unfinished_task = (~ if_reach_task_region) & unfinished_task
        # print(unfinished_task)
        task_state_decimal = np.dot(unfinished_task.T, 2 ** np.arange(unfinished_task.size)[::-1])[0]
        print(task_state_decimal)

        state_tensor = torch.cat((agent_pos_tensor, torch.tensor([[task_state_decimal]], dtype=torch.float32)), dim=1)
        state_tensor = state_tensor.repeat(all_action.shape[0], 1)
        state_all_action_tensor = torch.cat((state_tensor, all_action), dim=1)
        # print(state_all_action_tensor)
        q_value = Q_Network(state_all_action_tensor)
        action = torch.argmax(q_value).item() + 1
        print(q_value)
        print(action)
        x = (action - 1) % 3 - 1
        y = (action - 1) // 3 - 1
        v = accuracy * np.array([[x,y]], dtype=np.float32)
        # print(v)
        agent_pos += v
        print(agent_pos)
        agent1_x_list.append(agent_pos[0][0])
        agent1_y_list.append(agent_pos[0][1])

        step += 1
        if task_state_decimal == 0: done = True
        if step >= 1000: done = True

        if done:
            break


print(step)
fig, ax = plt.subplots()
# 绘图
ax.plot(agent1_x_list, agent1_y_list, marker='o', markersize=1, label="Agent 1")
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