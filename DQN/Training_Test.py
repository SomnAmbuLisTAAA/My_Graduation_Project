import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from NN_model import SequentialMultiLayerNN

agent_pos = np.array([ [25.0,25.0] ], dtype=np.float32)
task_pos = np.array([ [17.5,17.5] ], dtype=np.float32)

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

dis_agent2task = cdist(agent_pos,task_pos)
task_radius = np.array([ [0.5] ])
done = False
accuracy = 0.5
step = 0

agent1_x_list = []
agent1_y_list = []
agent1_x_list.append(agent_pos[0][0])
agent1_y_list.append(agent_pos[0][1])

Q_Network = SequentialMultiLayerNN(4,256,4,1)
Q_Network.eval()

path = "model/Deep_Q_network_01_03_No1.pth"
Q_Network.load_state_dict(torch.load(path, weights_only=True))

while not done:

    with torch.no_grad():
        all_action = torch.arange(1, 10).view(-1, 1)
        agent_pos_tensor = torch.from_numpy(agent_pos)
        state_tensor = torch.cat((agent_pos_tensor, torch.tensor([[1.0]], dtype=torch.float32)), dim=1)
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

        dis_agent2task = cdist(agent_pos, task_pos)
        done = (dis_agent2task <= task_radius)[0][0]
        step += 1
        if step >= 1000: done = True

print(step)
fig, ax = plt.subplots()
# 绘图
ax.plot(agent1_x_list, agent1_y_list, marker='o', markersize=1, label="Agent 1")
# ax.scatter(agent1_x_list, agent1_y_list, marker='o', s=1, label="Agent 1")

for center, radius in zip(obs_pos, obs_radius):
    circle = plt.Circle(center, radius, color='red', fill=True)
    ax.add_artist(circle)

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