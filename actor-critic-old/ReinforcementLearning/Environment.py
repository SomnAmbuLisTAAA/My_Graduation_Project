import copy
from collections import deque
import gym
from gym import spaces
import numpy as np
import random
import ReinforcementLearning.Agent as agent
from gym.envs.registration import register
import torch
from DeepLearning.NN_model import SequentialMultiLayerNN


class ContinuousMultiAgentEnv(gym.Env):
    def __init__(self, width=10.0, height=10.0, agent_list=None, obstacle_list=None, obstacle_radius=None, task_list=None,
                 task_radius=None, sample_time=0.01):
        """ 初始化连续多智能体环境 """

        super(ContinuousMultiAgentEnv, self).__init__()

        # 环境尺寸
        self.width = width
        self.height = height

        # 奖励设置
        self.reward_cross_border = -10
        self.reward_touch_obstacle = -10
        self.reward_crash_another_agent = -10
        self.reward_reach_task_region = 100

        # 奖励的获得情况
        self.times_cross_border = 0
        self.times_touch_obstacle = 0
        self.times_crash_another_agent = 0
        self.times_reach_task_region = 0

        # agent_group: 智能体对象列表
        self.agent_group = agent_list if agent_list else []

        # agent_positions: 各智能体位置
        # speed_limits: 各个智能体的速度范围列表，每个元素是 (min_x, max_x, min_y, max_y)
        # agent_radius: 各智能体尺寸
        if self.agent_group:

            self.agent_positions = []
            self.speed_limits = []
            self.agent_radius = []

            for ags in self.agent_group:
                self.agent_positions.append(ags.get_agent_pos().tolist())
                self.speed_limits.append(ags.get_speed_limits())
                self.agent_radius.append(ags.get_agent_radius())

        else :
            self.agent_positions = []
            self.speed_limits = []
            self.agent_radius = []

        # last_agent_positions: 各智能体上一step位置
        # agent_init_positions: 各智能体初始位置
        # num_agents: 智能体个数
        self.last_agent_positions = self.agent_positions
        self.agent_init_positions = self.agent_positions
        self.num_agents = len(self.agent_positions)

        # 存储状态
        self.state_memory = deque(maxlen=11)
        self.state_memory.append(np.array(self.agent_positions, dtype=np.float32).flatten())

        self.s1_batch = deque(maxlen=100)
        self.sn1_batch = deque(maxlen=100)

        # obstacle_positions: 障碍物位置列表
        # obstacle_radius: 障碍物半径列表
        self.obstacle_positions = obstacle_list if obstacle_list else []
        self.obstacle_radius = obstacle_radius if obstacle_list else []
        self.num_obstacles = len(self.obstacle_positions)

        # task_positions: 任务位置
        # task_radius: 任务区域半径
        self.task_positions = task_list if task_list else []
        self.task_radius = task_radius if task_radius else []
        self.num_tasks = len(self.task_positions)

        # sample_time: 执行一步所花时间
        self.sample_time = sample_time

        # 动作空间：每个智能体的动作为二维连续值（x和y方向的速度）
        low = np.array([[limits[0], limits[2]] for limits in self.speed_limits], dtype=np.float32)  # x 和 y 的最小速度
        high = np.array([[limits[1], limits[3]] for limits in self.speed_limits], dtype=np.float32)  # x 和 y 的最大速度
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 状态空间：每个智能体的位置为二维连续值
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_agents * 2,), dtype=np.float32),
            high=np.array([self.width, self.height] * self.num_agents, dtype=np.float32),  # 每个智能体的 (x, y) 范围
            dtype=np.float32
        )

        # 如果未指定智能体、障碍物或任务，则随机初始化
        self._initialize_positions()

    def _initialize_positions(self):
        """ 如果未指定智能体、障碍物或任务，则随机初始化 """
        if not self.agent_positions:
            self.agent_positions = [self._random_position() for _ in range(2)]
            self.last_agent_positions = self.agent_positions
            self.agent_init_positions = self.agent_positions
            self.num_agents = len(self.agent_positions)
        else: self.agent_positions = self.agent_init_positions

        if not self.speed_limits:
            self.speed_limits = [(-1,1,-1,1) for _ in range(self.num_agents)]

        if not self.agent_radius:
            self.agent_radius = [0.5 for _ in range(self.num_agents)]

        if not self.obstacle_positions:
            self.obstacle_positions = [self._random_position() for _ in range(5)]
            self.obstacle_radius = [0.5 for _ in range(5)]
            self.num_obstacles = len(self.obstacle_positions)

        if not self.task_positions:
            self.task_positions = [self._random_position() for _ in range(3)]
            self.task_radius = [0.5 for _ in range(3)]
            self.num_tasks = len(self.task_positions)

    def reset(self):
        """ 将环境位置恢复至初始状态 """
        self.times_cross_border = 0
        self.times_touch_obstacle = 0
        self.times_crash_another_agent = 0
        self.times_reach_task_region = 0

        self._initialize_positions()
        for ags in self.agent_group:
            ags.agent_init()
        observation = np.array(self.agent_positions, dtype=np.float32)
        info = {"reset_info": "initialization details"}
        return observation.flatten(), info

    def step(self, actions):
        """
        执行一步环境的状态更新。
        :param actions: 一个包含每个智能体动作的列表或数组，形状为 (num_agents, 2)，表示每个智能体在 x 和 y 方向的速度。
        :return: 下一状态、奖励、是否完成、调试信息
        """
        # print(self.agent_positions)

        undo_list = []
        rewards = []
        info = {
            "cross_border": [],
            "touch_obstacle": [],
            "crash_another_agent": [],
            "reach_task_region": []
        }

        self.last_agent_positions = copy.deepcopy(self.agent_positions)
        # print(self.last_agent_positions)

        # 遍历所有智能体，更新位置
        # actions 是一个形状为 (num_agents, 2) 的数组，表示所有智能体的动作
        # 示例：actions = np.array([[vx1, vy1], [vx2, vy2], ..., [vxN, vyN]])

        # 确保动作在允许的速度范围内
        # speed_limits 是一个形状为 (num_agents, 4) 的数组，表示每个智能体在 x 和 y 方向的速度限制
        # speed_limits[i] = [min_vx, max_vx, min_vy, max_vy]
        self.speed_limits = np.array(self.speed_limits, dtype=np.float32)
        clipped_actions = np.clip(actions, self.speed_limits[:, 0:1], self.speed_limits[:, 1:2])

        # 更新所有智能体的位置
        # self.agent_positions 是一个形状为 (num_agents, 2) 的数组，表示所有智能体的当前位置
        # sample_time 是时间间隔
        self.agent_positions += clipped_actions * self.sample_time

        # 遍历所有智能体，计算奖励
        for i in range(self.num_agents):

            reward = 0

            flag_cross_border = 0
            if (self.agent_positions[i][0] < self.agent_radius[i]
                    or self.agent_positions[i][0] > self.width - self.agent_radius[i]
                    or self.agent_positions[i][1] < self.agent_radius[i]
                    or self.agent_positions[i][1] > self.height - self.agent_radius[i]):
                flag_cross_border = 1
                self.times_cross_border += 1

            if flag_cross_border == 1:
                reward += self.reward_cross_border
                info["cross_border"].append(f"agent:{i+1}")
                undo_list.append(i)

            flag_touch_obstacle = 0
            for j in range(self.num_obstacles):

                if (np.linalg.norm( self.agent_positions[i] - np.array(self.obstacle_positions[j]) )
                        <= self.agent_radius[i] + self.obstacle_radius[j]):
                    flag_touch_obstacle = 1
                    info["touch_obstacle"].append(f"agent:{i+1}, obstacle:{j+1}")
                    reward += self.reward_touch_obstacle
                    self.times_touch_obstacle += 1

            if flag_touch_obstacle == 1:
                undo_list.append(i)

            flag_crash_another_agent = 0
            for j in range(self.num_agents):

                if i != j:
                    if (np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                            <= self.agent_radius[i] + self.agent_radius[j]):
                        flag_crash_another_agent = 1
                        info["crash_another_agent"].append(f"agent:{i+1}, agent:{j+1}")
                        reward += self.reward_crash_another_agent
                        self.times_crash_another_agent += 0.5

            if flag_crash_another_agent == 1:
                undo_list.append(i)

            for j in range(self.num_tasks):

                if (np.linalg.norm(self.agent_positions[i] - np.array(self.task_positions[j]))
                        <= self.task_radius[j]):
                    info["reach_task_region"].append(f"agent:{i+1}, task:{j+1}")
                    reward += self.reward_reach_task_region
                    self.times_reach_task_region += 1

            rewards.append(reward)
            self.agent_group[i].save_experience(reward)

        # print(self.last_agent_positions)
        # print(self.agent_positions)
        # print(undo_list)
        for i in undo_list:
            self.agent_positions[i] = self.last_agent_positions[i]

        observation = np.array(self.agent_positions, dtype=np.float32)
        self.state_memory.append(observation.flatten())

        if len(self.state_memory) == self.state_memory.maxlen:
            self.s1_batch.append(self.state_memory[0])
            self.sn1_batch.append(self.state_memory[-1])

        # print(self.state_memory)
        terminated = False
        truncated = False  # 你可以根据时间限制设置是否截断
        return observation, rewards, terminated, truncated, info

    def render(self):
        """渲染环境状态。"""
        print(f"Agent Positions: {self.agent_positions}")
        print(f"Last Agent Positions: {self.last_agent_positions}")
        # print(f"Obstacle Positions: {self.obstacle_positions}")
        # print(f"Task Positions: {self.task_positions}")

    def train(self):

        if len(self.state_memory) == self.state_memory.maxlen:

            s1 = self.state_memory[0]
            s1 = torch.tensor(s1, dtype=torch.float32).view(1, -1)
            # print(self.state_memory)

            sn1 = self.state_memory[-1]
            sn1 = torch.tensor(sn1, dtype=torch.float32).view(1, -1)

            for agent in self.agent_group:
                agent.calculate_critic_loss(s1, sn1)
                agent.update_critic_network()
                agent.calculate_actor_loss(s1)
                agent.update_actor_network()

    def train_v2(self):
        if len(self.s1_batch) == self.s1_batch.maxlen:

            s1_array = np.array(list(self.s1_batch))
            s1_batch = torch.tensor(s1_array)
            sn1_array = np.array(list(self.sn1_batch))
            sn1_batch = torch.tensor(sn1_array)

            # print(s1_batch)

            self.s1_batch.clear()
            self.sn1_batch.clear()

            for agent in self.agent_group:
                agent.upgrade_network(s1_batch,sn1_batch)

    def save_model(self,strs):
        for agent,str in zip(self.agent_group,strs):
            agent.save_model(str[0],str[1])

    def load_model(self,strs):
        for agent,str in zip(self.agent_group,strs):
            agent.load_model(str[0],str[1])

    def get_loss(self):
        loss = []
        for agent in self.agent_group:
            loss.append(agent.get_average_loss())
        return loss

    def get_policy_actions(self):
        actions = []
        state = torch.tensor(np.array(self.agent_positions, dtype=np.float32).flatten()).unsqueeze(0)
        # print(state)
        for agent in self.agent_group:
            actions.append(agent.policy_act(state).tolist())
        return actions

    def _random_position(self):
        """随机生成在环境内的连续位置。"""
        return np.array([random.uniform(0, self.width), random.uniform(0, self.height)])

    def return_reward_times(self):
        return [self.times_cross_border, self.times_touch_obstacle,
                self.times_crash_another_agent, self.times_reach_task_region]

register(
    id='ContinuousMultiAgent-v0',
    entry_point='ReinforcementLearning.Environment:ContinuousMultiAgentEnv',
    max_episode_steps=100000000,
)







class ContinuousMultiAgentEnv_v2(gym.Env):
    def __init__(self, width=10.0, height=10.0, agent_list=None, obstacle_list=None, obstacle_radius=None, task_list=None,
                 task_radius=None, sample_time=0.01):
        """ 初始化连续多智能体环境_v2 """

        super(ContinuousMultiAgentEnv_v2, self).__init__()

        # 环境尺寸
        self.width = width
        self.height = height

        # 奖励设置
        self.reward_cross_border = -2.0
        self.reward_touch_obstacle = -2.0
        self.reward_crash_another_agent = -12.0
        self.reward_reach_task_region = 100.0

        # 奖励的获得情况
        self.times_cross_border = 0
        self.times_touch_obstacle = 0
        self.times_crash_another_agent = 0
        self.times_reach_task_region = 0

        # agent_group: 智能体对象列表
        self.agent_group = agent_list if agent_list else []

        # agent_positions: 各智能体位置
        # speed_limits: 各个智能体的速度范围列表，每个元素是 (min_x, max_x, min_y, max_y)
        # agent_radius: 各智能体尺寸
        if self.agent_group:

            self.agent_positions = []
            self.speed_limits = []
            self.agent_radius = []

            for ags in self.agent_group:
                self.agent_positions.append(ags.get_agent_pos().tolist())
                self.speed_limits.append(ags.get_speed_limits())
                self.agent_radius.append(ags.get_agent_radius())

        else :
            self.agent_positions = []
            self.speed_limits = []
            self.agent_radius = []

        # last_agent_positions: 各智能体上一step位置
        # agent_init_positions: 各智能体初始位置
        # num_agents: 智能体个数
        self.last_agent_positions = self.agent_positions
        self.agent_init_positions = self.agent_positions
        self.num_agents = len(self.agent_positions)

        # 存储状态
        self.state_memory = deque(maxlen=11)
        self.state_memory.append(np.array(self.agent_positions, dtype=np.float32).flatten())

        self.s1_batch = deque(maxlen=100)
        self.sn1_batch = deque(maxlen=100)

        # obstacle_positions: 障碍物位置列表
        # obstacle_radius: 障碍物半径列表
        # num_obstacles: 障碍物个数
        self.obstacle_positions = obstacle_list if obstacle_list else []
        self.obstacle_radius = obstacle_radius if obstacle_list else []
        self.num_obstacles = len(self.obstacle_positions)

        # task_positions: 任务位置
        # task_radius: 任务区域半径
        self.task_positions = task_list if task_list else []
        self.task_radius = task_radius if task_radius else []
        self.num_tasks = len(self.task_positions)

        # sample_time: 执行一次决策的间隔时间
        self.sample_time = sample_time

        # 动作空间：每个智能体的动作为二维连续值（x和y方向的速度）
        low = np.array([[limits[0], limits[2]] for limits in self.speed_limits], dtype=np.float32)  # x 和 y 的最小速度
        high = np.array([[limits[1], limits[3]] for limits in self.speed_limits], dtype=np.float32)  # x 和 y 的最大速度
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 状态空间：每个智能体的位置为二维连续值
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_agents * 2,), dtype=np.float32),
            high=np.array([self.width, self.height] * self.num_agents, dtype=np.float32),  # 每个智能体的 (x, y) 范围
            dtype=np.float32
        )

        # 初始化中心critic网络
        input_dim = 4 * self.num_agents
        self.central_critic_network = SequentialMultiLayerNN(input_dim,128,2,self.num_agents)

        # 如果未指定智能体、障碍物或任务，则随机初始化
        self._initialize_positions()

    def _initialize_positions(self):
        """ 如果未指定智能体、障碍物或任务，则随机初始化 """
        if not self.agent_positions:
            self.agent_positions = [self._random_position() for _ in range(2)]
            self.last_agent_positions = self.agent_positions
            self.agent_init_positions = self.agent_positions
            self.num_agents = len(self.agent_positions)
        else: self.agent_positions = self.agent_init_positions

        if not self.speed_limits:
            self.speed_limits = [(-1,1,-1,1) for _ in range(self.num_agents)]

        if not self.agent_radius:
            self.agent_radius = [0.5 for _ in range(self.num_agents)]

        if not self.obstacle_positions:
            self.obstacle_positions = [self._random_position() for _ in range(5)]
            self.obstacle_radius = [0.5 for _ in range(5)]
            self.num_obstacles = len(self.obstacle_positions)

        if not self.task_positions:
            self.task_positions = [self._random_position() for _ in range(3)]
            self.task_radius = [0.5 for _ in range(3)]
            self.num_tasks = len(self.task_positions)

    def reset(self):
        """ 将环境位置恢复至初始状态 """
        self.times_cross_border = 0
        self.times_touch_obstacle = 0
        self.times_crash_another_agent = 0
        self.times_reach_task_region = 0

        self._initialize_positions()
        for ags in self.agent_group:
            ags.agent_init()
        observation = np.array(self.agent_positions, dtype=np.float32)
        info = {"reset_info": "initialization details"}
        return observation.flatten(), info













import gym
import ReinforcementLearning.Agent as agent
import ReinforcementLearning.Agent_v2 as agent_v2
import numpy as np
import time
import ReinforcementLearning.Environment
import matplotlib.pyplot as plt

# 使用环境
if __name__ == "__main__":

    start_time = time.time()
    # 设置初始位置
    agent1 = agent_v2.Actor_Critic.simplest_ac_agent(2,2,0.5,-10.0,10.0,-10.0,10.0,1.0,1.0)
    agent2 = agent_v2.Actor_Critic.simplest_ac_agent(2,2,0.5,-10.0,10.0,-10.0,10.0,5.0,5.0)
    agent_group = [agent1,agent2]

    obstacles = [np.array([10.0, 10.0]), np.array([20.0, 20.0]), np.array([20.0, 5.0]), np.array([5.0, 20.0]), np.array([15.0, 20.0])]
    obstacle_radius = [0.5, 1.0, 0.5, 0.5, 1.0]
    tasks = [np.array([8.0, 8.0]), np.array([15.0, 2.0])]
    task_radius = [1.0,1.0]

    env = gym.make('ContinuousMultiAgent-v0',width=30.0,height=30.0, agent_list=agent_group, obstacle_list=obstacles,
                   obstacle_radius=obstacle_radius, task_list=tasks, task_radius=task_radius, sample_time=0.05)

    env.reset()

    agent1_pos_list = []
    agent2_pos_list = []

    agent_pos = env.agent_group[0].get_agent_pos()
    agent1_pos_list.append(tuple(agent_pos.tolist()))
    agent_pos = env.agent_group[1].get_agent_pos()
    agent2_pos_list.append(tuple(agent_pos.tolist()))

    rounds = 1000000
    for round in range(rounds):
        action = np.random.uniform(-10.0, 10.0, (len(agent_group), 2))
        observation, rewards, terminated, truncated, info = env.step(action)
        agent1_pos_list.append(observation[0])
        agent2_pos_list.append(observation[1])

        if (round + 1) % 100 == 0:
            print(f"Rounds: [{100 * (round + 1) / rounds}%]")

        #observation, rewards, terminated, truncated, info = env.step(action)
        #env.render()
        #print(info)
        #print("\n")

    times = env.return_reward_times()
    print(f"times_cross_border:{times[0]}")
    print(f"times_touch_obstacle:{times[1]}")
    print(f"times_crash_another_agent:{times[2]}")
    print(f"times_reach_task_region:{times[3]}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time:.6f} 秒")

    # print(agent1_pos_list)

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
