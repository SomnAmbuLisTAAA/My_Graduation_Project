from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time

from Neural_network_model import SequentialMultiLayerNN

device = 'cuda'

critic_learning_rate = 1e-4
weight_decay = 1e-3

central_critic_network = SequentialMultiLayerNN(9,256,4,1)
central_critic_network_optimizer = optim.AdamW(central_critic_network.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)