import numpy as np
dis_agent2task = np.array([[1,2,3,4],[4,3,2,1]])
if_reach_task_region=np.array([[True,False,False,True], [True,False,False,True]])
unfinished_tasks = np.array([[1],[1],[1],[1]])
result = np.zeros_like(if_reach_task_region, dtype=bool)

for col in range(if_reach_task_region.shape[1]):
    # 找到当前列中为 True 的行索引
    true_indices = np.where(if_reach_task_region[:, col])[0]

    if len(true_indices) > 0:  # 如果有 True 元素
        # 找到这些行对应的 self.dis_agent2task 值
        col_values = dis_agent2task[true_indices, col]
        # 找到最小值对应的行索引
        min_index = true_indices[np.argmin(col_values)]
        # 在结果矩阵中保留这个索引对应的 True
        result[min_index, col] = True

print(result)
print(np.dot(result,unfinished_tasks))