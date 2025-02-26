import numpy as np
import pulp

class TaskAssignmentSolver:
    def __init__(self, n, m):
        """
        预先定义优化问题结构
        :param n: 智能体数量
        :param m: 任务数量
        """
        self.n = n
        self.m = m
        self.problem = pulp.LpProblem("Task_Assignment", pulp.LpMinimize)

        # 定义二进制变量 x_ij，表示智能体 i 是否执行任务 j
        self.x = np.array([[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(m)] for i in range(n)])

        # 约束1：每个任务至少分配给一个智能体
        for j in range(m):
            self.problem += pulp.lpSum(self.x[:, j]) >= 1, f"Task_{j}_assigned"

    def solve(self, cost_matrix):
        """
        更新目标函数并求解
        :param cost_matrix: np.array 形状为 (n, m)，表示 n 个智能体到 m 个任务的代价矩阵
        :return: 任务分配方案 [(agent, task)], 每个智能体的成本数组, 最小总成本
        """
        # 更新目标函数：最小化总成本
        self.problem.setObjective(pulp.lpSum(cost_matrix[i, j] * self.x[i, j] for i in range(self.n) for j in range(self.m)))

        # 解决问题时关闭日志
        self.problem.solve(pulp.PULP_CBC_CMD(msg=False))

        # 获取分配结果，避免 Python 循环，通过向量化实现
        assignment = np.array([(i, j) for i in range(self.n) for j in range(self.m) if pulp.value(self.x[i, j]) == 1])

        # 计算每个智能体的成本，使用 NumPy 向量化
        agent_costs = np.sum(cost_matrix[assignment[:, 0], assignment[:, 1]], axis=0)

        return agent_costs