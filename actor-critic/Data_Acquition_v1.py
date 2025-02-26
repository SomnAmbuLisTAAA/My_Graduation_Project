import numpy as np
import torch

from Environment_v2 import MultiAgentTrainingEnv_2D_v2

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

env = MultiAgentTrainingEnv_2D_v2(30.0,30.0,agent_init_pos,agent_radius,obs_pos,obs_radius,
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

trajectory_reward_matrix = np.empty((final_size, 1), dtype=np.float32)
o1_1_matrix = np.empty((final_size, 20) ,dtype=np.float32)
o2_1_matrix = np.empty((final_size, 20) ,dtype=np.float32)
o1_n1_matrix = np.empty((final_size, 20) ,dtype=np.float32)
o2_n1_matrix = np.empty((final_size, 20) ,dtype=np.float32)
action1_1_matrix = np.empty((final_size, 2), dtype=np.float32)
action2_1_matrix = np.empty((final_size, 2), dtype=np.float32)

line = 0

while line != final_size:

    observation_1 ,observation_n1, action_1_new_line, trajectory_reward = env.step()

    if not observation_1.size == 0:

        # print(agent_action_1_new_row)

        trajectory_reward_matrix[line] = trajectory_reward
        o1_1_matrix[line] = observation_1[0]
        o2_1_matrix[line] = observation_1[1]
        o1_n1_matrix[line] = observation_n1[0]
        o2_n1_matrix[line] = observation_n1[1]
        action1_1_matrix[line] = action_1_new_line[0]
        action2_1_matrix[line] = action_1_new_line[1]
        line += 1

        '''print(trajectory_reward)
        print(observation_1)
        print(observation_n1)'''

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

        if 0.25 < (line + 1) / final_size <=0.50 and (observation_1[0][4] == 0 or observation_1[0][4] == 1):

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

        if 0.50 < (line + 1) / final_size <= 0.75 and (observation_1[0][4] == 0 or observation_1[0][4] == 2):

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

        if 0.75 < (line + 1) / final_size and observation_1[0][4] != 3:
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
action1_1_tensor = torch.from_numpy(action1_1_matrix)
action2_1_tensor = torch.from_numpy(action2_1_matrix)

trajectory_reward_tensor = trajectory_reward_tensor.to(device)
o1_1_tensor = o1_1_tensor.to(device)
o2_1_tensor = o2_1_tensor.to(device)
o1_n1_tensor = o1_n1_tensor.to(device)
o2_n1_tensor = o2_n1_tensor.to(device)
action1_1_tensor = action1_1_tensor.to(device)
action2_1_tensor = action2_1_tensor.to(device)

path = 'Experience/DataSet_02_09_No2_tensors.pth'
torch.save({'trajectory_reward_tensor': trajectory_reward_tensor, 'o1_1_tensor': o1_1_tensor,
            'o2_1_tensor': o2_1_tensor, 'o1_n1_tensor':o1_n1_tensor, 'o2_n1_tensor': o2_n1_tensor,
            'action1_1_tensor': action1_1_tensor, 'action2_1_tensor': action2_1_tensor}, path)
print("张量存储完成")

print(trajectory_reward_tensor)
print(o1_1_tensor)
print(o2_1_tensor)
print(o1_n1_tensor)
print(o2_n1_tensor)
print(action1_1_tensor)
print(action2_1_tensor)

