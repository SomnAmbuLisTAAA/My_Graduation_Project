import numpy as np




trajectory_reward = 0.0
immediate_reward_memory_np_array = np.array([[1,2,3],[4,5,6]])
gamma_powers = np.array([[1,2,3]])
trajectory_reward = np.sum((immediate_reward_memory_np_array * gamma_powers),axis=1).reshape(1,-1)
print(trajectory_reward)