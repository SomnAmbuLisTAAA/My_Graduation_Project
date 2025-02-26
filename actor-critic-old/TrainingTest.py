from DeepLearning.NN_model import SequentialMultiLayerNN, SequentialMultiLayerNN_with_scaled_tanh
import numpy as np
import torch
import matplotlib.pyplot as plt

agent_init_pos = np.array([ [5.0,10.0],[25.0,20.0] ])
agent_radius = np.array([ [0.2],[0.2] ])

obs_pos = np.array([ [5.0,5.0], [5.0,15.0], [5.0,25.0],
                     [10.0,5.0],[10.0,15.0],[10.0,25.0],
                     [15.0,5.0],[15.0,15.0],[15.0,25.0],
                     [20.0,5.0],[20.0,15.0],[20.0,25.0],
                     [25.0,5.0],[25.0,15.0],[25.0,25.0] ])
obs_radius = np.array([ [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0],
                        [1.0],[1.0],[1.0] ])

task_pos = np.array([ [15.0,20.0],[15.0,10.0] ])
task_radius = np.array([ [0.5],[0.5] ])

actor_network = SequentialMultiLayerNN_with_scaled_tanh(2,128,3,2,10,0.1)

path = ["DeepLearning/model/12_13/actor_network_04.pth",
            "DeepLearning/model/12_13/central_critic_network_04.pth"]

actor_network.load_state_dict(torch.load(path[0], weights_only=True))
sample_time = 0.05
steps = 1000

agent1_x_list = []
agent1_y_list = []
agent2_x_list = []
agent2_y_list = []

actor_network.eval()
agent_pos = agent_init_pos
agent_pos_tensor = torch.from_numpy(agent_pos).float()
v_tensor = torch.zeros_like(agent_pos_tensor)
print("v_tensor:\n",v_tensor)
print('agent_pos_tensor:\n', agent_pos_tensor,'\n')

agent1_x_list.append(agent_pos_tensor[0][0].item())
agent1_y_list.append(agent_pos_tensor[0][1].item())
agent2_x_list.append(agent_pos_tensor[1][0].item())
agent2_y_list.append(agent_pos_tensor[1][1].item())

for step in range(steps):
        v_tensor = actor_network(agent_pos_tensor)
        agent_pos_tensor += v_tensor * sample_time
        agent1_x_list.append(agent_pos_tensor[0][0].item())
        agent1_y_list.append(agent_pos_tensor[0][1].item())
        agent2_x_list.append(agent_pos_tensor[1][0].item())
        agent2_y_list.append(agent_pos_tensor[1][1].item())
        print("v_tensor:\n", v_tensor)
        print('agent_pos_tensor:\n',agent_pos_tensor,'\n')

fig, ax = plt.subplots()

# 绘图
ax.plot(agent1_x_list, agent1_y_list, marker='o', markersize=2, label="Agent 1")
ax.plot(agent2_x_list, agent2_y_list, marker='o', markersize=2, label="Agent 2")
'''
ax.scatter(agent1_x_list, agent1_y_list, marker='o', s=0.1, label="Agent 1")
ax.scatter(agent2_x_list, agent2_y_list, marker='o', s=0.1, label="Agent 2")
'''

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
# s1_a1_tensor_mean = torch.tensor([[ 1.4879e+01,  1.4456e+01,  1.5459e+01,  1.4465e+01, -3.7845e-03, -4.4250e-03,  6.3986e-03, -6.3457e-03]])
# s1_a1_tensor_std = torch.tensor([[8.4972, 8.6988, 8.6760, 8.6307, 5.7772, 5.7720, 5.7703, 5.7714]])

x_y_mean_tensor = torch.tensor([[1.4879e+01, 1.4456e+01], [1.5459e+01, 1.4465e+01]])
x_y_std_tensor = torch.tensor([[8.4972, 8.6988],[8.6760, 8.6307]])

v_mean = torch.tensor([[-3.7845e-03, -4.4250e-03],[6.3986e-03, -6.3457e-03]])
V_std = torch.tensor([[5.7772, 5.7720],[5.7703, 5.7714]])

with torch.no_grad():
    actor_network.eval()
    agent_pos = agent_init_pos

    agent_pos_tensor = torch.from_numpy(agent_pos).float()
    print('agent_pos_tensor:\n', agent_pos_tensor)
    normalized_agent_pos = (agent_pos_tensor - x_y_mean_tensor) / x_y_std_tensor
    # print("normalized_agent_pos:\n", normalized_agent_pos)
    v_tensor = torch.zeros_like(agent_pos_tensor)
    print("v_tensor:\n",v_tensor)
    # print('agent_pos_tensor:\n',agent_pos_tensor)

    for step in range(steps):
        v_tensor = actor_network(normalized_agent_pos)
        agent_pos_tensor += v_tensor * sample_time
        normalized_agent_pos = (agent_pos_tensor - x_y_mean_tensor) / x_y_std_tensor
        # print("v_tensor:\n", v_tensor)
        print('agent_pos_tensor:\n',agent_pos_tensor)
'''