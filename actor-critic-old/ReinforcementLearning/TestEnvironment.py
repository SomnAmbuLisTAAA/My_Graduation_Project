import numpy as np
import ReinforcementLearning.Agent as agent
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from collections import deque

class ContinuousEnv:
    def __init__(self, agent_group, obstacle_radius, width, height, obstacle_positions=None, goal_position=None, sample_time = 0.01):
        self.width = width  # 环境的宽度
        self.height = height  # 环境的高度
        self.reward_cross_border = -10
        self.reward_forbidden_area = -10
        self.reward_agent_crash = -10
        self.reward_target_area = 100

        # 如果没有传入障碍物位置，则设置为空列表
        self.obstacle_positions = obstacle_positions if obstacle_positions is not None else []
        # self.obstacle_radius = obstacle_radius if obstacle_radius is not None else []
        self.obstacle_radius = obstacle_radius

        # 如果没有传入目标位置，则设置默认目标位置
        self.goal_position = goal_position if goal_position is not None else []

        # 设置智能体
        self.agent_group = agent_group if agent_group is not None else []

        self.sample_time = sample_time
        self.agent_pos = []

        self.cb: int = 0
        self.ob: int = 0
        self.ac: int = 0
        self.reach: int = 0

    def reset(self):
        for i in range(len(self.agent_group)):
            self.agent_group[i].agent_init()
            x ,y = self.agent_group[i].get_agent_pos()
            if x<= 0 or x >= self.width or y <= 0 or y >= self.height:
                self.agent_group[i].agent_default_init()

    def upgrade_pos(self):
        for i in range(len(self.agent_group)):
            self.agent_group[i].explore_act()
            self.agent_group[i].upgrade_pos(self.sample_time)
            
    def upgrade_pos_behavior_policy(self):
        for i in range(len(self.agent_group)):
            self.agent_group[i].policy_act()
            self.agent_group[i].upgrade_pos(self.sample_time)
            # print(self.agent_group[i].get_agent_pos())

    def get_reward(self):

        reward = []
        undo_list = []

        for i in range(len(self.agent_group)):

            agent_reward: int = 0
            agent_pos = self.agent_group[i].get_agent_pos()

            agent_radius = self.agent_group[i].get_agent_radius()
            # print(agent_radius)
            x,y = agent_pos.tolist()

            if x<= agent_radius or x >= self.width - agent_radius or y <= agent_radius or y >= self.height - agent_radius:
                undo_list.append(i)
                agent_reward = agent_reward + self.reward_cross_border
                reward.append([i+1,'cb',-10,self.agent_group[i].get_agent_pos()])
                self.cb = self.cb + 1

            for j in range(len(self.obstacle_positions)):
                if np.linalg.norm(agent_pos - np.array(self.obstacle_positions[j])) <= agent_radius + self.obstacle_radius:
                    # print(agent_pos)
                    # print(np.array(self.obstacle_positions[j]))
                    # print(np.linalg.norm(agent_pos - np.array(self.obstacle_positions[j])))
                    undo_list.append(i)
                    agent_reward = agent_reward + self.reward_forbidden_area
                    reward.append([i+1,j+1,'ob',-10])
                    self.ob = self.ob + 1
                    break

            flag: int = 0

            for j in range(len(self.agent_group)):

                if i != j:
                    another_agent_pos = self.agent_group[j].get_agent_pos()
                    if np.linalg.norm(agent_pos - another_agent_pos) <= 2 * agent_radius:
                        flag = 1
                        undo_list.append(j)
                        agent_reward = agent_reward + self.reward_agent_crash
                        reward.append([i+1,j+1,'ac',-10])
                        self.ac = self.ac + 1

            if flag == 1:
                undo_list.append(i)

            for goal_pos in self.goal_position:
                if np.linalg.norm(agent_pos - np.array(goal_pos)) <= 0.5 * agent_radius:
                    agent_reward = agent_reward + self.reward_target_area
                    reward.append([i + 1, 'reach', 100])
                    self.reach = self.reach + 1

            self.agent_group[i].write_reward(agent_reward)

        for i in undo_list:
            self.agent_group[i].undo_pos()

        #print(f"Reward List: {reward}")

    def render(self):
        # 可视化环境状态
        self.agent_pos.clear()
        for ag in self.agent_group:
            agent_position = ag.get_agent_pos()
            self.agent_pos.append(tuple(agent_position.tolist()))
        print(f"Agent Position: {self.agent_pos}")
        print(f"Goal Position: {self.goal_position}")
        print(f"Obstacle Positions: {self.obstacle_positions}")

    def return_agent_position(self):
        self.agent_pos.clear()
        for ag in self.agent_group:
            agent_position = ag.get_agent_pos()
            self.agent_pos.append(tuple(agent_position.tolist()))
            # agent_position = ag.get_last_pos()
            # self.agent_pos.append(tuple(agent_position.tolist()))
        print(f"Agent Position: {self.agent_pos}")

    def print_negative_reward_time(self):
        print(f"ob: {self.ob}")
        print(f"cb: {self.cb}")
        print(f"ac: {self.ac}")
        print(f"reach: {self.reach}")

    def print_reward(self):
        reward = []
        for ags in self.agent_group:
            agent_reward = ags.get_reward()
            reward.append(agent_reward)
        print(reward)

    #def store_experience(self):

class ContinuousMultiAgentTrainingEnv_2D_v0:
    def __init__(self, width=10.0, height=10.0, agent_init_pos=None, agent_radius=None, obstacle_pos=None,
                 obstacle_radius=None, task_pos=None, task_radius=None, sample_time=0.01, multi_step = 10):

        # 环境参数
        self.width = width
        self.height = height
        self.step = 0
        self.gamma = 0.95

        # 奖励信息
        self.reward_collision_border = -2
        self.reward_collision_obstacle = -2
        self.reward_collision_another_agent = -2
        self.reward_reach_task_region = 100

        # 初始化agent信息
        self.agent_init_pos = agent_init_pos.copy()
        self.agent_pos = agent_init_pos.copy()
        self.agent_last_pos = agent_init_pos.copy()
        self.agent_v = np.zeros_like(self.agent_pos, dtype=float)
        self.agent_radius = agent_radius.copy()

        self.left_border = agent_radius.copy()
        self.right_border = self.width - self.agent_radius
        self.top_border = self.height - self.agent_radius
        self.bottom_border = agent_radius.copy()

        self.undo_list = np.zeros((self.agent_pos.shape[0], 1))

        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        # 初始化障碍信息
        self.obstacle_pos = obstacle_pos.copy()
        self.obstacle_radius = obstacle_radius.copy()

        # 初始化任务信息
        self.task_pos = task_pos.copy()
        self.task_radius = task_radius.copy().T
        # print(self.task_radius)
        # print(self.agent_pos.shape[0])
        self.task_radius = np.tile(self.task_radius, (self.agent_pos.shape[0], 1))
        # print(self.task_radius)

        # 初始化agent间的距离、各agent与各障碍物间的距离、各agent与任务点间的距离
        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        self.radius_agent_plus_obstacle = self.agent_radius + self.obstacle_radius.T
        self.radius_agent_plus_agent = self.agent_radius + self.agent_radius.T
        # print(self.radius_agent_plus_agent)
        np.fill_diagonal(self.radius_agent_plus_agent, -1.0)
        # print(self.radius_agent_plus_agent)

        self.accumulate_reward = np.zeros((self.agent_pos.shape[0], 1),dtype=float)
        self.immediate_reward = self.accumulate_reward.copy()

        self.sample_time = sample_time

        self.state_memory = deque(maxlen=multi_step + 1)
        self.state_memory.append(self.agent_pos.flatten())
        self.action_memory = deque(maxlen=multi_step)
        self.immediate_reward_memory = deque(maxlen=multi_step)
        self.trajectory_reward = 0.0
        self.gamma_powers = self.gamma ** np.arange(multi_step)
        # print(self.gamma_powers)


    def reset(self):
        self.agent_pos = self.agent_init_pos.copy()
        self.agent_last_pos = self.agent_init_pos.copy()
        self.agent_v[:] = 0.0

        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        self.accumulate_reward[:] = 0.0
        self.immediate_reward[:] = 0.0

        self.step = 0

        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        self.state_memory.clear()
        self.state_memory.append(self.agent_pos.flatten())
        self.action_memory.clear()
        self.immediate_reward_memory.clear()
        self.trajectory_reward = 0.0

    def parameter_check_agent(self):
        print("agent_pos:\n",self.agent_pos)
        print("agent_last_pos:\n",self.agent_last_pos)
        print("agent_init_pos:\n",self.agent_init_pos)
        print("agent_radius:\n",self.agent_radius)
        print("agent_v:\n",self.agent_v)

    def parameter_check_basic(self):
        print("width:\n",self.width)
        print("height:\n",self.height)
        print("obstacle_pos:\n",self.obstacle_pos)
        print("obstacle_radius:\n",self.obstacle_radius)
        print("task_pos:\n",self.task_pos)
        print("task_radius:\n",self.task_radius)
        print("sample_time:\n",self.sample_time)
        print("radius_agent_plus_obstacle:\n",self.radius_agent_plus_obstacle)
        print("radius_agent_plus_agent:\n",self.radius_agent_plus_agent)

    def parameter_check_dis(self):
        print("dis_agent2agent:\n",self.dis_agent2agent)
        print("dis_agent2obstacle:\n",self.dis_agent2obstacle)
        print("dis_agent2task:\n",self.dis_agent2task)

    def parameter_check_reward(self):
        print("step:\n",self.step)
        print("accumulate_reward:\n",self.accumulate_reward)
        print("immediate_reward:\n",self.immediate_reward)

    def parameter_check_all(self):
        self.parameter_check_basic()
        self.parameter_check_agent()
        self.parameter_check_dis()
        self.parameter_check_reward()

    def get_reward_times(self):
        print('times_collision_another_agent:\n',self.times_collision_another_agent)
        print('times_collision_obstacle:\n',self.times_collision_obstacle)
        print('times_collision_border:\n',self.times_collision_border)
        print('times_reach_task_region:\n',self.times_reach_task_region)

    def explore_action(self, vx_min, vx_max, vy_min, vy_max):
        n = self.agent_v.shape[0]
        vx = np.random.uniform(vx_min, vx_max, size=n)
        vy = np.random.uniform(vy_min, vy_max, size=n)
        self.agent_v = np.column_stack((vx, vy))
        return self.agent_v

    def update_pos_by_action(self):
        self.step += 1
        self.agent_last_pos = self.agent_pos.copy()
        self.agent_pos += self.sample_time * self.agent_v
        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)
        return self.agent_pos

    def calculate_reward(self):

        self.immediate_reward[:] = 0.0

        if_collision_another_agent = self.dis_agent2agent <= self.radius_agent_plus_agent
        # print(if_collision_another_agent)
        if_collision_another_agent = np.sum(if_collision_another_agent, axis=1).reshape(-1, 1)
        # print("if_collision_another_agent:\n",if_collision_another_agent)
        self.immediate_reward += self.reward_collision_another_agent * if_collision_another_agent
        # print(self.immediate_reward)
        self.times_collision_another_agent += if_collision_another_agent

        if_collision_obstacle = self.dis_agent2obstacle <= self.radius_agent_plus_obstacle
        if_collision_obstacle = np.sum(if_collision_obstacle, axis=1).reshape(-1, 1)
        # print("if_collision_obstacle:\n",if_collision_obstacle)
        self.immediate_reward += self.reward_collision_obstacle * if_collision_obstacle
        self.times_collision_obstacle += if_collision_obstacle

        agent_x = self.agent_pos[:, 0].reshape(-1, 1)
        agent_y = self.agent_pos[:, 1].reshape(-1, 1)
        # print(agent_x)
        # print(agent_y)
        # print(self.top_border)
        # print(self.bottom_border)
        # print((agent_x <= self.left_border) | (agent_x >= self.right_border))
        if_collision_border = ((agent_x <= self.left_border) | (agent_x >= self.right_border)
                               | (agent_y <= self.bottom_border) | (agent_y >= self.top_border))
        # print('if_collision_border:\n',if_collision_border)
        self.immediate_reward += self.reward_collision_border * if_collision_border
        self.times_collision_border += if_collision_border

        # print(self.dis_agent2task)
        # print(self.task_radius)
        if_reach_task_region = self.dis_agent2task <= self.task_radius
        # print(self.dis_agent2task)
        if_reach_task_region = np.sum(if_reach_task_region, axis=1).reshape(-1, 1)
        # print('if_reach_task_region:\n',if_reach_task_region)
        self.immediate_reward += self.reward_reach_task_region * if_reach_task_region
        self.times_reach_task_region += if_reach_task_region

        min_dis_agent2task = np.min(self.dis_agent2task, axis=1, keepdims=True)
        # print(min_dis_agent2task)
        self.immediate_reward += -min_dis_agent2task
        # print(self.immediate_reward)

        self.undo_list = if_collision_another_agent + if_collision_obstacle + if_collision_border
        self.undo_list[self.undo_list != 0] = 1
        # print('undo_list:\n',self.undo_list)
        # print("agent_pos:\n",self.agent_pos)
        # print("agent_last_pos:\n", self.agent_last_pos)

        self.accumulate_reward += ( self.immediate_reward * ( self.gamma**( self.step - 1 ) ) )
        self.agent_pos[self.undo_list.squeeze() == 1] = self.agent_last_pos[self.undo_list.squeeze() == 1]
        # print('修正后的agent_pos:\n',self.agent_pos)

        return self.immediate_reward, self.accumulate_reward, self.agent_pos

    def get_experience(self):
        self.state_memory.append(self.agent_pos.flatten())
        self.action_memory.append(self.agent_v.flatten())
        self.immediate_reward_memory.append(np.sum(self.immediate_reward).item())
        # print(self.immediate_reward_memory[-1])

        if len(self.immediate_reward_memory) == self.immediate_reward_memory.maxlen:
            immediate_reward_memory_np_array = np.array(list(self.immediate_reward_memory))
            '''print(immediate_reward_memory_np_array)
            print(self.gamma_powers)
            print(immediate_reward_memory_np_array * self.gamma_powers)'''
            self.trajectory_reward = np.sum(immediate_reward_memory_np_array * self.gamma_powers)
            # print(self.trajectory_reward)
            return np.hstack((self.state_memory[0], self.action_memory[0])),  self.state_memory[-1], self.trajectory_reward

        else: return np.hstack((self.state_memory[0], self.action_memory[0])), np.array([]), self.trajectory_reward



class ContinuousMultiAgentTrainingEnv_2D_v1:
    def __init__(self, width=10.0, height=10.0, agent_init_pos=None, agent_radius=None, obstacle_pos=None,
                 obstacle_radius=None, task_pos=None, task_radius=None, sample_time=0.01, multi_step = 10, gamma=0.95):

        # 环境参数
        self.done = False
        self.width = width
        self.height = height
        self.step = 0
        self.gamma = gamma

        # 奖励信息
        self.reward_collision_border = -2
        self.reward_collision_obstacle = -2
        self.reward_collision_another_agent = -2
        self.reward_reach_task_region = 100

        # 初始化agent信息
        self.agent_init_pos = agent_init_pos.copy()
        self.agent_pos = agent_init_pos.copy()
        self.agent_last_pos = agent_init_pos.copy()
        self.agent_v = np.zeros_like(self.agent_pos, dtype=float)
        self.agent_radius = agent_radius.copy()

        self.left_border = agent_radius.copy()
        self.right_border = self.width - self.agent_radius
        self.top_border = self.height - self.agent_radius
        self.bottom_border = agent_radius.copy()

        self.undo_list = np.zeros((self.agent_pos.shape[0], 1))

        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        # 初始化障碍信息
        self.obstacle_pos = obstacle_pos.copy()
        self.obstacle_radius = obstacle_radius.copy()

        # 初始化任务信息
        self.task_pos = task_pos.copy()
        self.task_radius = task_radius.copy().T
        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1), dtype=int)
        # print("self.unfinished_tasks:\n",self.unfinished_tasks,"\n\n\n")
        # print(self.task_radius)
        # print(self.agent_pos.shape[0])
        self.task_radius = np.tile(self.task_radius, (self.agent_pos.shape[0], 1))
        # print(self.task_radius)

        # 初始化agent间的距离、各agent与各障碍物间的距离、各agent与任务点间的距离
        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        self.radius_agent_plus_obstacle = self.agent_radius + self.obstacle_radius.T
        self.radius_agent_plus_agent = self.agent_radius + self.agent_radius.T
        # print(self.radius_agent_plus_agent)
        np.fill_diagonal(self.radius_agent_plus_agent, -1.0)
        # print(self.radius_agent_plus_agent)

        self.accumulate_reward = np.zeros((self.agent_pos.shape[0], 1),dtype=float)
        self.immediate_reward = self.accumulate_reward.copy()

        self.sample_time = sample_time

        self.agent_pos_memory = deque(maxlen=multi_step + 1)
        self.agent_pos_memory.append(self.agent_pos.flatten())

        self.task_state_memory = deque(maxlen=multi_step + 1)
        self.task_state_memory.append(self.unfinished_tasks.T)

        self.agent_action_memory = deque(maxlen=multi_step)
        self.immediate_reward_memory = deque(maxlen=multi_step)
        self.trajectory_reward = 0.0
        self.gamma_powers = self.gamma ** np.arange(multi_step)
        # print(self.gamma_powers)


    def reset(self):
        self.done = False
        self.agent_pos = self.agent_init_pos.copy()
        self.agent_last_pos = self.agent_init_pos.copy()
        self.agent_v[:] = 0.0

        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1))

        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        self.accumulate_reward[:] = 0.0
        self.immediate_reward[:] = 0.0

        self.step = 0

        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1)).astype(int)
        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        self.agent_pos_memory.clear()
        self.task_state_memory.clear()
        self.agent_pos_memory.append(self.agent_pos.flatten())
        self.task_state_memory.append(self.unfinished_tasks.T)
        self.agent_action_memory.clear()
        self.immediate_reward_memory.clear()
        self.trajectory_reward = 0.0

    def reset_random_pos(self):
        self.done = False
        while True:
            x_coords = np.random.uniform(self.left_border.flatten(), self.right_border.flatten())  # 随机生成 x 坐标
            y_coords = np.random.uniform(self.top_border.flatten(), self.bottom_border.flatten())  # 随机生成 y 坐标
            self.agent_pos = np.column_stack((x_coords, y_coords))

            self.dis_agent2agent = cdist(self.agent_pos, self.agent_pos)
            self.dis_agent2obstacle = cdist(self.agent_pos, self.obstacle_pos)

            if_collision_obstacle = self.dis_agent2obstacle <= self.radius_agent_plus_obstacle
            if_collision_another_agent = self.dis_agent2agent <= self.radius_agent_plus_agent

            if not np.any(if_collision_obstacle) and not np.any(if_collision_another_agent):
                print("self.agent_pos:\n",self.agent_pos)
                '''print("self.dis_agent2agent:\n",self.dis_agent2agent)
                print("self.dis_agent2obstacle:\n",self.dis_agent2obstacle)'''
                break

        self.dis_agent2task = cdist(self.agent_pos, self.task_pos)

        self.agent_last_pos = self.agent_pos.copy()
        self.agent_init_pos = self.agent_pos.copy()
        self.agent_v[:] = 0.0

        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1))

        self.accumulate_reward[:] = 0.0
        self.immediate_reward[:] = 0.0

        self.step = 0

        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1)).astype(int)
        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        self.agent_pos_memory.clear()
        self.task_state_memory.clear()
        self.agent_pos_memory.append(self.agent_pos.flatten())
        self.task_state_memory.append(self.unfinished_tasks.T)
        self.agent_action_memory.clear()
        self.immediate_reward_memory.clear()
        self.trajectory_reward = 0.0

    def parameter_check_agent(self):
        print("agent_pos:\n",self.agent_pos)
        print("agent_last_pos:\n",self.agent_last_pos)
        print("agent_init_pos:\n",self.agent_init_pos)
        print("agent_radius:\n",self.agent_radius)
        print("agent_v:\n",self.agent_v)

    def parameter_check_basic(self):
        print("width:\n",self.width)
        print("height:\n",self.height)
        print("obstacle_pos:\n",self.obstacle_pos)
        print("obstacle_radius:\n",self.obstacle_radius)
        print("task_pos:\n",self.task_pos)
        print("task_radius:\n",self.task_radius)
        print("sample_time:\n",self.sample_time)
        print("radius_agent_plus_obstacle:\n",self.radius_agent_plus_obstacle)
        print("radius_agent_plus_agent:\n",self.radius_agent_plus_agent)

    def parameter_check_dis(self):
        print("dis_agent2agent:\n",self.dis_agent2agent)
        print("dis_agent2obstacle:\n",self.dis_agent2obstacle)
        print("dis_agent2task:\n",self.dis_agent2task)

    def parameter_check_reward(self):
        print("step:\n",self.step)
        print("accumulate_reward:\n",self.accumulate_reward)
        print("immediate_reward:\n",self.immediate_reward)

    def parameter_check_all(self):
        self.parameter_check_basic()
        self.parameter_check_agent()
        self.parameter_check_dis()
        self.parameter_check_reward()

    def get_reward_times(self):
        print('times_collision_another_agent:\n',self.times_collision_another_agent)
        print('times_collision_obstacle:\n',self.times_collision_obstacle)
        print('times_collision_border:\n',self.times_collision_border)
        print('times_reach_task_region:\n',self.times_reach_task_region)

    def explore_action(self, vx_min, vx_max, vy_min, vy_max):
        n = self.agent_v.shape[0]
        vx = np.random.uniform(vx_min, vx_max, size=n)
        vy = np.random.uniform(vy_min, vy_max, size=n)
        self.agent_v = np.column_stack((vx, vy))
        return self.agent_v

    def update_pos_by_action(self):
        self.step += 1
        self.agent_last_pos = self.agent_pos.copy()
        self.agent_pos += self.sample_time * self.agent_v
        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)
        return self.agent_pos

    def calculate_reward(self):

        self.immediate_reward[:] = 0.0

        if_collision_another_agent = self.dis_agent2agent <= self.radius_agent_plus_agent
        # print(if_collision_another_agent)
        if_collision_another_agent = np.sum(if_collision_another_agent, axis=1).reshape(-1, 1)
        # print("if_collision_another_agent:\n",if_collision_another_agent)
        self.immediate_reward += self.reward_collision_another_agent * if_collision_another_agent
        # print(self.immediate_reward)
        self.times_collision_another_agent += if_collision_another_agent

        if_collision_obstacle = self.dis_agent2obstacle <= self.radius_agent_plus_obstacle
        if_collision_obstacle = np.sum(if_collision_obstacle, axis=1).reshape(-1, 1)
        # print("if_collision_obstacle:\n",if_collision_obstacle)
        self.immediate_reward += self.reward_collision_obstacle * if_collision_obstacle
        self.times_collision_obstacle += if_collision_obstacle

        agent_x = self.agent_pos[:, 0].reshape(-1, 1)
        agent_y = self.agent_pos[:, 1].reshape(-1, 1)
        # print(agent_x)
        # print(agent_y)
        # print(self.top_border)
        # print(self.bottom_border)
        # print((agent_x <= self.left_border) | (agent_x >= self.right_border))
        if_collision_border = ((agent_x <= self.left_border) | (agent_x >= self.right_border)
                               | (agent_y <= self.bottom_border) | (agent_y >= self.top_border))
        # print('if_collision_border:\n',if_collision_border)
        self.immediate_reward += self.reward_collision_border * if_collision_border
        self.times_collision_border += if_collision_border

        # print(self.dis_agent2task)
        # print(self.task_radius)
        if_reach_task_region = self.dis_agent2task <= self.task_radius
        # print('if_reach_task_region:\n',if_reach_task_region)

        '''
        # 创建一个与 if_reach_task_region 形状相同的全 False 矩阵
        result = np.zeros_like(if_reach_task_region, dtype=bool)
        # 对每一列进行操作
        for col in range(if_reach_task_region.shape[1]):
            # 找到当前列中为 True 的行索引
            true_indices = np.where(if_reach_task_region[:, col])[0]
            if len(true_indices) > 0:  # 如果有 True 元素
                # 找到这些行对应的 self.dis_agent2task 值
                col_values = self.dis_agent2task[true_indices, col]
                # 找到最小值对应的行索引
                min_index = true_indices[np.argmin(col_values)]
                # 在结果矩阵中保留这个索引对应的 True
                result[min_index, col] = True
        '''

        if_task_finished = np.any(if_reach_task_region, axis=0).reshape(-1, 1)
        # print("if_task_finished:\n",if_task_finished)

        if_reach_task_region = np.dot(if_reach_task_region, self.unfinished_tasks)
        # if_reach_task_region = np.dot(result, self.unfinished_tasks)
        # print('if_reach_task_region:\n',if_reach_task_region)

        # print(self.dis_agent2task)
        # if_reach_task_region = np.sum(if_reach_task_region, axis=1).reshape(-1, 1)
        # print('if_reach_task_region:\n', if_reach_task_region)
        self.immediate_reward += self.reward_reach_task_region * if_reach_task_region
        self.times_reach_task_region += if_reach_task_region

        # 更新unfinished_tasks
        self.unfinished_tasks = ( ~ if_task_finished) & self.unfinished_tasks
        # print("self.unfinished_tasks:\n",self.unfinished_tasks,"\n\n")
        self.done = np.all(self.unfinished_tasks == 0)

        if not self.done:
            mask = self.unfinished_tasks.flatten() == 1  # 将列向量转换为一维数组
            # 挑选出 dis_agent2task 中对应的列
            dis_agent2unfinished_task = self.dis_agent2task[:, mask]  # 按列索引
            # print("self.dis_agent2task:\n",self.dis_agent2task)
            # print("dis_agent2unfinished_task:\n",dis_agent2unfinished_task)
            min_dis_agent2unfinished_task = np.min(dis_agent2unfinished_task, axis=1, keepdims=True)
            # print("min_dis_agent2unfinished_task:\n",min_dis_agent2unfinished_task)
            self.immediate_reward += -min_dis_agent2unfinished_task
            # print(self.immediate_reward)

        self.undo_list = if_collision_another_agent + if_collision_obstacle + if_collision_border
        self.undo_list[self.undo_list != 0] = 1
        # print('undo_list:\n',self.undo_list)
        # print("agent_pos:\n",self.agent_pos)
        # print("agent_last_pos:\n", self.agent_last_pos)

        self.accumulate_reward += ( self.immediate_reward * ( self.gamma**( self.step - 1 ) ) )
        self.agent_pos[self.undo_list.squeeze() == 1] = self.agent_last_pos[self.undo_list.squeeze() == 1]
        # print('修正后的agent_pos:\n',self.agent_pos)

        return self.immediate_reward, self.accumulate_reward, self.agent_pos

    def get_experience(self):
        self.agent_pos_memory.append(self.agent_pos.flatten())
        self.task_state_memory.append(self.unfinished_tasks.T)
        self.agent_action_memory.append(self.agent_v.flatten())
        self.immediate_reward_memory.append(np.sum(self.immediate_reward).item())
        # print(self.immediate_reward_memory[-1])
        # print(self.task_state_memory)

        if len(self.immediate_reward_memory) == self.immediate_reward_memory.maxlen:
            immediate_reward_memory_np_array = np.array(list(self.immediate_reward_memory))
            '''print(immediate_reward_memory_np_array)
            print(self.gamma_powers)
            print(immediate_reward_memory_np_array * self.gamma_powers)'''
            self.trajectory_reward = np.sum(immediate_reward_memory_np_array * self.gamma_powers)
            # print(self.trajectory_reward)
            return (self.agent_pos_memory[0], self.agent_action_memory[0], self.agent_pos_memory[-1], self.trajectory_reward,
                    self.task_state_memory[0], self.task_state_memory[-1])

        else: return (self.agent_pos_memory[0], self.agent_action_memory[0], np.array([]), self.trajectory_reward,
                      self.task_state_memory[0], np.array([]))

    def if_done(self):
        return self.done

    def return_unfinished_tasks(self):
        return self.unfinished_tasks