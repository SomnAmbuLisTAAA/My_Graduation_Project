import numpy as np
from scipy.spatial.distance import cdist
from collections import deque

class MultiAgentTrainingEnv_2D_v2:

    def __init__(self, width=10.0, height=10.0, agent_init_pos=None, agent_radius=None, obstacle_pos=None,
                 obstacle_radius=None, task_pos=None, task_radius=None, accuracy=0.1, multi_step=10, gamma=0.99):
        """
        2D 离散多智能体训练环境
        Attributes:
            width (int): 环境宽度。
            height (int): 环境高度。
            agent_init_pos (np.array): 各智能体初始位置，列向量。
            agent_radius (np.array): 各智能体半径，列向量。
            obstacle_pos (np.ndarray): 障碍物位置，列向量。
            obstacle_radius (np.array): 障碍物半径，列向量。
            task_pos (np.array): 任务位置，列向量。
            task_radius (np.array): 任务半径，列向量。
            accuracy (float): 环境细分精度，默认值为 0.1。
            multi_step (int): 多步采集参数，与 TD_target 相关，默认值为 10 步。
            gamma (float): 折扣因子，默认值为 0.99。
        """

        self.done = False           # self.done：标志位，任务是否被全部完成

        self.width = width          # self.width：环境宽度
        self.height = height        # self.height：环境高度
        self.step_num = 0           # self.step_num：已经执行的步数
        self.gamma = gamma          # self.gamma：折扣因子

        self.reward_collision_border = -10          # self.reward_collision_border：碰撞边界获得的reward
        self.reward_collision_obstacle = -10        # self.reward_collision_obstacle：碰撞障碍物获得的reward
        self.reward_collision_another_agent = -10    # self.reward_collision_another_agent：与其他agent相撞获得的reward
        self.reward_reach_task_region = 100          # self.reward_reach_task_region：首次完成任务获得的reward
        self.reward_finish_task = 0                  # self.reward_finish_task：完成某个任务后持续获得的reward
        self.reward_all_tasks_finished = 0           # self.reward_all_tasks_finished：所有任务都完成后获得的reward

        # 初始化agent信息，均按行存储为一个矩阵
        # 示例：np.array([ [agent_1_info], [agent_2_info],..., [agent_N_info] ])
        self.agent_init_pos = agent_init_pos.copy()                 # self.agent_init_pos：agent的初始位置
        self.agent_pos = agent_init_pos.copy()                      # self.agent_pos：存储agent的实时位置
        self.agent_last_pos = agent_init_pos.copy()                 # self.agent_last_pos：存储agent上一时刻的位置
        self.agent_v = np.zeros_like(self.agent_pos, dtype=float)   # self.agent_v：agent的当前速度
        self.agent_radius = agent_radius.copy()                     # self.agent_radius：各agent的半径

        # 对于各agent而言的活动边界，均按行存储为一个矩阵
        # 示例见上，agent信息
        self.left_border = agent_radius.copy()                  # self.left_border：各agent x坐标所能达到的最小值
        self.right_border = self.width - self.agent_radius      # self.right_border：各agent x坐标所能达到的最大值
        self.bottom_border = agent_radius.copy()                # self.bottom_border：各agent y坐标所能达到的最小值
        # print(self.bottom_border)
        self.top_border = self.height - self.agent_radius       # self.top_border：各agent y坐标所能达到的最大值

        # self.undo_list：若采取某action后，agent与边界、障碍物或其他agent相撞，需要回到原位置，用于存储某步后哪些agent需要回到上一坐标
        # 只存储0、1的列向量，若agent i需要回到上一位置，则i行会标记为1，否则为0
        self.undo_list = np.zeros((self.agent_pos.shape[0], 1))

        # 存储一episode下(self.done由False到True)，各agent获得的各种奖励的次数，np.array列向量
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
        self.task_radius = np.tile(self.task_radius, (self.agent_pos.shape[0], 1))
        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1), dtype=int)
        # self.unfinished_tasks：某一任务是否未完成的标志位，若i任务未完成，则i行为1
        # print("self.unfinished_tasks:\n",self.unfinished_tasks,"\n\n\n")
        # print(self.task_radius)
        # print(self.agent_pos.shape[0])
        # print(self.task_radius)

        # 初始化agent间的距离、各agent与各障碍物间的距离、各agent与任务点间的距离
        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        # 用于判断agent与障碍、agent与agent之间是否会相撞
        self.radius_agent_plus_obstacle = self.agent_radius + self.obstacle_radius.T
        # self.radius_agent_plus_obstacle：
        # 矩阵，内容为[[ra1+ro1, ra1+ro2,..., ra1+rom], [ra2+ro1, ra2+ro2,..., ra2+rom],..., [ran+ro1, ran+ro2,..., ran+rom]]
        self.radius_agent_plus_agent = self.agent_radius + self.agent_radius.T
        # print(self.radius_agent_plus_agent)
        # self.radius_agent_plus_agent：
        # 矩阵，内容为[[ra1+ra1, ra1+ra2,..., ra1+ran], [ra2+ra1, ra2+ra2,..., ra2+ran],..., [ran+ra1, ran+ra2,..., ran+ran]]
        np.fill_diagonal(self.radius_agent_plus_agent, -1.0)    # 将主对角线上的元素置为-1，因为agent与自身不用考虑是否相撞

        # print(self.radius_agent_plus_agent)
        # print(self.radius_agent_plus_agent)

        self.accumulate_reward = np.zeros((self.agent_pos.shape[0], 1),dtype=float)     # 存储各agent获得的累计奖励
        self.immediate_reward = self.accumulate_reward.copy()                           # 存储各agent获得的即时奖励

        self.accuracy = accuracy  # 细分精度

        self.agent_pos_memory = deque(maxlen=multi_step + 1)        # 记录一个multi_step上各agent的位置
        self.agent_pos_memory.append(self.agent_pos.copy())

        self.task_state_memory = deque(maxlen=multi_step + 1)       # 记录一个multi_step上的任务状态
        self.task_state_memory.append(self.unfinished_tasks.T.copy())

        self.agent_action_memory = deque(maxlen=multi_step)         # 记录一个multi_step上agent采取的action
        self.immediate_reward_memory = deque(maxlen=multi_step)     # 记录一个multi_step上各agent获取的即时奖励

        self.dis_agent2task_memory = deque(maxlen=multi_step + 1)
        self.dis_agent2agent_memory = deque(maxlen=multi_step + 1)
        self.dis_agent2obstacle_memory = deque(maxlen=multi_step + 1)
        self.dis_agent2task_memory.append(self.dis_agent2task.copy())
        self.dis_agent2agent_memory.append(self.dis_agent2agent.copy())
        self.dis_agent2obstacle_memory.append(self.dis_agent2obstacle.copy())

        self.trajectory_reward = 0.0                                # 存储一个multi_step的路径回报
        self.gamma_powers = self.gamma ** np.arange(multi_step)     # 用于计算trajectory_reward，存储衰减因子的各次方
        # trajectory_reward = immediate_reward(1) + gamma^1*immediate_reward(2) + ... + gamma^(n-1)*immediate_reward(n)
        # self.parameter_check_all()

    def reset(self, mode=1, new_agent_init_pos=None):
        """
        重置训练环境
        重置agent位置，重置环境信息与记录
        """
        self.done = False
        self.step_num = 0

        if mode == 1:
            self.agent_pos = self.agent_init_pos.copy()
            self.agent_last_pos = self.agent_init_pos.copy()

        if mode == 2 and new_agent_init_pos is not None:

            assert isinstance(new_agent_init_pos, np.ndarray), "输入必须是一个 NumPy 数组，列向量。"
            assert new_agent_init_pos.shape[1] == self.agent_pos.shape[1], "坐标维数错误"

            self.agent_pos = new_agent_init_pos.copy()
            self.agent_init_pos = new_agent_init_pos.copy()
            self.agent_last_pos = new_agent_init_pos.copy()

        self.agent_v = np.zeros_like(self.agent_pos, dtype=float)

        self.undo_list = np.zeros((self.agent_pos.shape[0], 1))

        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))

        self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1), dtype=int)

        self.dis_agent2agent = cdist(self.agent_pos,self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos,self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos,self.task_pos)

        self.accumulate_reward = np.zeros((self.agent_pos.shape[0], 1),dtype=float)
        self.immediate_reward = self.accumulate_reward.copy()

        self.agent_pos_memory.clear()
        self.task_state_memory.clear()
        self.agent_action_memory.clear()
        self.immediate_reward_memory.clear()
        self.dis_agent2task_memory.clear()
        self.dis_agent2agent_memory.clear()
        self.dis_agent2obstacle_memory.clear()
        self.trajectory_reward = 0.0

        self.agent_pos_memory.append(self.agent_pos.copy())
        self.task_state_memory.append(self.unfinished_tasks.T.copy())
        self.dis_agent2task_memory.append(self.dis_agent2task.copy())
        self.dis_agent2agent_memory.append(self.dis_agent2agent.copy())
        self.dis_agent2obstacle_memory.append(self.dis_agent2obstacle.copy())
        # self.parameter_check_all()

    def parameter_check_basic(self):
        print("done:",self.done)
        print("step:", self.step_num)

        print("width:", self.width)
        print("height:", self.height)
        print("gamma:", self.gamma)
        print("accuracy:", self.accuracy)
        print("multi_step:",self.agent_action_memory.maxlen)
        print("reward_collision_border:",self.reward_collision_border)
        print("reward_collision_obstacle:", self.reward_collision_obstacle)
        print("reward_collision_another_agent:", self.reward_collision_another_agent)
        print("reward_reach_task_region:", self.reward_finish_task)
        print("reward_all_tasks_finished:", self.reward_all_tasks_finished)

        print("radius_agent_plus_obstacle:\n",self.radius_agent_plus_obstacle)
        print("radius_agent_plus_agent:\n", self.radius_agent_plus_agent)
        print("task_radius:\n",self.task_radius)

    def parameter_check_task(self):
        print("task_pos:\n", self.task_pos)
        print("unfinished_tasks:\n",self.unfinished_tasks)
        print("task_radius:\n", self.task_radius)

    def parameter_check_obstacle(self):
        print("obstacle_pos:\n", self.obstacle_pos)
        print("obstacle_radius:\n", self.obstacle_radius)

    def parameter_check_agent(self):
        print("agent_pos:\n", self.agent_pos)
        print("agent_last_pos:\n", self.agent_last_pos)
        print("agent_init_pos:\n", self.agent_init_pos)
        print("agent_radius:\n", self.agent_radius)
        print("agent_v:\n", self.agent_v)

    def parameter_check_dis(self):
        print("dis_agent2agent:\n", self.dis_agent2agent)
        print("dis_agent2obstacle:\n", self.dis_agent2obstacle)
        print("dis_agent2task:\n", self.dis_agent2task)

    def parameter_check_reward(self):
        print("accumulate_reward:\n", self.accumulate_reward)
        print("immediate_reward:\n", self.immediate_reward)

    def get_reward_times(self):
        print('times_collision_another_agent:\n', self.times_collision_another_agent)
        print('times_collision_obstacle:\n', self.times_collision_obstacle)
        print('times_collision_border:\n', self.times_collision_border)
        print('times_reach_task_region:\n', self.times_reach_task_region)

    def get_memory_data(self):
        print("agent_pos_memory:\n",self.agent_pos_memory)
        print("agent_action_memory:\n", self.agent_action_memory)
        print("task_state_memory:\n", self.task_state_memory)
        print("immediate_reward_memory:\n",self.immediate_reward_memory)

    def parameter_check_all(self):
        self.parameter_check_basic()
        self.parameter_check_agent()
        self.get_memory_data()
        self.parameter_check_task()
        self.parameter_check_obstacle()
        self.parameter_check_dis()
        self.parameter_check_reward()
        self.get_reward_times()

    def explore_action(self):
        sampling_values = [-1, -0.5, 0, 0.5, 1]
        num_samples = self.agent_pos.shape[0]
        vx = np.random.choice(sampling_values, size=num_samples).reshape(-1, 1)
        vy = np.random.choice(sampling_values, size=num_samples).reshape(-1, 1)
        self.agent_v = np.hstack((vx, vy)) * self.accuracy
        # print(self.agent_v)

    def update_agent_pos_by_v(self):
        self.step_num += 1
        self.agent_last_pos = self.agent_pos.copy()
        # print(self.agent_last_pos)
        self.agent_pos += self.agent_v
        # print(self.agent_pos)
        self.dis_agent2agent = cdist(self.agent_pos, self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos, self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos, self.task_pos)

    def calculate_reward(self):

        self.immediate_reward[:] = 0.0

        if_collision_another_agent = self.dis_agent2agent <= self.radius_agent_plus_agent
        '''print(self.dis_agent2agent)
        print(self.radius_agent_plus_agent)
        print(if_collision_another_agent)'''
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
        if_reach_task_region = self.dis_agent2task <= self.task_radius              # 获取agent是否抵达任务区域，对应元素标记为True

        if_task_finish_first_time = np.dot(if_reach_task_region, self.unfinished_tasks)
        self.immediate_reward += self.reward_reach_task_region * if_task_finish_first_time
        self.times_reach_task_region += if_task_finish_first_time

        if_task_finished = np.any(if_reach_task_region, axis=0).reshape(-1, 1)  # 每列元素做或运算，得到哪一个task被完成，标记为True
        self.unfinished_tasks = (~ if_task_finished) & self.unfinished_tasks  # 更新unfinished_tasks

        num_finished_task = np.sum(self.unfinished_tasks == 0)
        num_finished_task_column_vector = np.full((self.agent_pos.shape[0], 1), num_finished_task)

        self.immediate_reward += self.reward_finish_task * num_finished_task_column_vector

        self.done = np.all(self.unfinished_tasks == 0)
        # print("self.dis_agent2task:\n", self.dis_agent2task)

        if not self.done:

            mask = self.unfinished_tasks.flatten() == 1  # 将列向量转换为一维数组
            # 挑选出 dis_agent2task 中对应的列
            dis_agent2unfinished_task = self.dis_agent2task[:, mask]  # 按列索引
            # print("self.dis_agent2task:\n",self.dis_agent2task)
            # print("dis_agent2unfinished_task:\n",dis_agent2unfinished_task)

            agent_costs = np.zeros((self.agent_pos.shape[0], 1))  # 初始化智能体的代价为 0
            min_row_indices = np.argmin(dis_agent2unfinished_task, axis=0)  # 找到每列最小值的行索引
            # 遍历每列任务
            for col in range(dis_agent2unfinished_task.shape[1]):
                row = min_row_indices[col]  # 获取该任务对应的最近的智能体
                agent_costs[row][0] += dis_agent2unfinished_task[row][col]  # 累加代价到对应智能体
            self.immediate_reward += -1.0 * agent_costs

            '''min_dis_agent2unfinished_task = np.min(dis_agent2unfinished_task, axis=1, keepdims=True)
            # print("min_dis_agent2unfinished_task:\n",min_dis_agent2unfinished_task)
            # min_dis_agent2unfinished_task = 8.96257338 * np.log(min_dis_agent2unfinished_task + 0.73169835) + 2.79979086
            # print("min_dis_agent2unfinished_task:\n", min_dis_agent2unfinished_task)
            self.immediate_reward += -0.5 * min_dis_agent2unfinished_task'''

            '''# print(self.dis_agent2task)
            # 挑选出 dis_agent2task 中对应的列
            mask = self.unfinished_tasks.flatten() == 1  # 将列向量转换为一维数组
            dis_agent2unfinished_task = self.dis_agent2task[:, mask]  # 按列索引
            # print(dis_agent2unfinished_task)
            dis_agent2selected_task = np.zeros((self.agent_pos.shape[0], 1))
            # 扁平化矩阵并进行排序，按从小到大的顺序
            sorted_flattened_indices = np.argsort(dis_agent2unfinished_task, axis=None)
            # 创建一个掩码矩阵，初始时为全部True
            mask = np.ones_like(dis_agent2unfinished_task, dtype=bool)
            # 获取每个选中位置的行列索引
            for idx in sorted_flattened_indices:
                row, col = np.unravel_index(idx, dis_agent2unfinished_task.shape)
                # 如果对应行列还未被选中，选择该位置
                if mask[row, col]:
                    dis_agent2selected_task[row, :] = dis_agent2unfinished_task[row, col]
                    mask[row, :] = False  # 禁止选取该行
                    mask[:, col] = False  # 禁止选取该列
            self.immediate_reward += -0.5 * dis_agent2selected_task'''

        else: self.immediate_reward += self.reward_all_tasks_finished

        # self.undo_list = if_collision_another_agent | if_collision_obstacle | if_collision_border
        self.undo_list = if_collision_another_agent | if_collision_border
        '''print('undo_list:\n',self.undo_list)
        print("agent_pos:\n",self.agent_pos)
        print("agent_last_pos:\n", self.agent_last_pos)'''

        self.accumulate_reward += (self.immediate_reward * (self.gamma ** (self.step_num - 1)))
        self.agent_pos[self.undo_list.squeeze() == 1] = self.agent_last_pos[self.undo_list.squeeze() == 1]
        '''print('修正后的agent_pos:\n',self.agent_pos)
        print("self.immediate_reward:\n",self.immediate_reward,"\n")'''
        # print("self.immediate_reward:\n",self.immediate_reward)
        self.dis_agent2agent = cdist(self.agent_pos, self.agent_pos)
        self.dis_agent2obstacle = cdist(self.agent_pos, self.obstacle_pos)
        self.dis_agent2task = cdist(self.agent_pos, self.task_pos)

    def save_experience(self):
        # print(self.agent_pos)
        self.agent_pos_memory.append(self.agent_pos.copy())
        # print(self.agent_pos_memory)
        self.task_state_memory.append(self.unfinished_tasks.T.copy())
        self.agent_action_memory.append(self.agent_v.copy())
        # print(self.agent_action_memory)
        self.immediate_reward_memory.append(np.sum(self.immediate_reward).item())

        self.dis_agent2task_memory.append(self.dis_agent2task.copy())
        self.dis_agent2agent_memory.append(self.dis_agent2agent.copy())
        self.dis_agent2obstacle_memory.append(self.dis_agent2obstacle.copy())
        # print(self.immediate_reward_memory[-1])
        # print(self.task_state_memory)
        # print("self.unfinished_tasks:\n",self.unfinished_tasks)

        if len(self.immediate_reward_memory) == self.immediate_reward_memory.maxlen:
            immediate_reward_memory_np_array = np.array(list(self.immediate_reward_memory))
            # print(immediate_reward_memory_np_array)
            # print(self.gamma_powers)
            # print(immediate_reward_memory_np_array * self.gamma_powers)
            self.trajectory_reward = np.sum(immediate_reward_memory_np_array * self.gamma_powers)
            return True

        else: return False

    def step(self):

        """"""

        '''if self.done:
            print("episode finish")
            self.get_reward_times()
            self.reset_by_init_pos()
            print("env reset finished\n")'''

        self.explore_action()
        self.update_agent_pos_by_v()
        self.calculate_reward()
        if_trajectory_reward = self.save_experience()

        if if_trajectory_reward:
            task_state_1_decimal = np.array([np.dot(self.task_state_memory[0], 2 ** np.arange(self.task_state_memory[0].size)[::-1])])
            task_state_n1_decimal = np.array([np.dot(self.task_state_memory[-1], 2 ** np.arange(self.task_state_memory[-1].size)[::-1])])


            observation_1_matrix = np.hstack((self.agent_pos_memory[0], self.dis_agent2task_memory[0],
                                              np.tile(task_state_1_decimal, (self.agent_pos.shape[0], 1)),
                                              self.dis_agent2agent_memory[0], self.dis_agent2obstacle_memory[0]))

            observation_n1_matrix = np.hstack((self.agent_pos_memory[-1], self.dis_agent2task_memory[-1],
                                              np.tile(task_state_n1_decimal, (self.agent_pos.shape[0], 1)),
                                              self.dis_agent2agent_memory[-1], self.dis_agent2obstacle_memory[-1]))

            return observation_1_matrix, observation_n1_matrix, self.agent_action_memory[0], self.trajectory_reward

        else: return np.array([]), np.array([]), np.array([]), np.array([])

    def if_done(self):
        return self.done

    def reset_task(self,reset_index_list = None):

        if reset_index_list  is not None:
            self.unfinished_tasks[reset_index_list, 0] = 1
        else:
            self.unfinished_tasks = np.ones((self.task_pos.shape[0], 1), dtype=int)
        self.clear_memory()

    def change_agent_pos(self, agent_pos=None):
        """改变智能体位置，不改变任务状态，清空信息缓存"""
        assert isinstance(agent_pos, np.ndarray), "输入必须是一个 NumPy 数组，列向量。"
        assert agent_pos.shape[1] == self.agent_pos.shape[1], "坐标维数错误"
        self.agent_pos = agent_pos.copy()
        self.agent_last_pos = self.agent_pos.copy()
        self.clear_memory()

    def clear_memory(self):

        self.accumulate_reward = np.zeros((self.agent_pos.shape[0], 1),dtype=float)
        self.immediate_reward = self.accumulate_reward.copy()
        self.agent_v = np.zeros_like(self.agent_pos, dtype=float)

        self.agent_pos_memory.clear()
        self.task_state_memory.clear()
        self.agent_action_memory.clear()
        self.immediate_reward_memory.clear()
        self.dis_agent2task_memory.clear()
        self.dis_agent2agent_memory.clear()
        self.dis_agent2obstacle_memory.clear()
        self.trajectory_reward = 0.0

        self.agent_pos_memory.append(self.agent_pos.copy())
        self.task_state_memory.append(self.unfinished_tasks.T.copy())
        self.dis_agent2task_memory.append(self.dis_agent2task.copy())
        self.dis_agent2agent_memory.append(self.dis_agent2agent.copy())
        self.dis_agent2obstacle_memory.append(self.dis_agent2obstacle.copy())

        self.times_collision_border = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_obstacle = np.zeros((self.agent_pos.shape[0], 1))
        self.times_collision_another_agent = np.zeros((self.agent_pos.shape[0], 1))
        self.times_reach_task_region = np.zeros((self.agent_pos.shape[0], 1))
