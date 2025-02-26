import numpy as np
import pulp

def min_cost_assignment(cost_matrix):
    """
    使用整数线性规划求解任务分配问题
    :param cost_matrix: np.array, 形状为 (n, m)，表示 n 个智能体到 m 个任务的代价矩阵
    :return: 任务分配方案 [(agent, task)], 最小总成本
    """
    n, m = cost_matrix.shape  # n 个智能体, m 个任务

    # 创建一个最小化问题
    problem = pulp.LpProblem("Task_Assignment", pulp.LpMinimize)

    # 定义二进制变量 x_ij，表示智能体 i 是否执行任务 j
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(m)] for i in range(n)]

    # 目标函数：最小化总成本
    problem += pulp.lpSum(cost_matrix[i, j] * x[i][j] for i in range(n) for j in range(m))

    # 约束1：每个任务至少分配给一个智能体
    for j in range(m):
        problem += pulp.lpSum(x[i][j] for i in range(n)) >= 1, f"Task_{j}_assigned"

    # 求解
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # 获取分配结果
    assignment = [(i, j) for i in range(n) for j in range(m) if pulp.value(x[i][j]) == 1]

    # 计算每个智能体的成本
    agent_costs = np.zeros((n,1))  # 创建全零列向量
    for i, j in assignment:
        agent_costs[i] += cost_matrix[i, j]  # 记录智能体 i 执行任务 j 的成本

    return agent_costs


# 示例矩阵 (行: 智能体, 列: 任务)
cost_matrix = np.array([[1, 2, 3],
                        [2, 3, 5],
                        [3, 5, 6]])

cost = min_cost_assignment(cost_matrix)

print("成本:", cost)