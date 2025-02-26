import NN_training_v1
from collections import deque

rounds = 5000
alpha_min = 0.2
alpha_max = 0.6
beta_min = 0.4
beta_max = 1.0
factor = 3.5

max_length = 10
actor_loss_deque = deque(maxlen=max_length)
critic_loss_deque = deque(maxlen=max_length)

NN_training_v1.update_priorities_of_buffer()

NN_training_v1.central_critic_network.train()
NN_training_v1.actor_network.train()

for round in range(rounds):

    actor_loss, critic_loss = NN_training_v1.step()

    actor_loss_deque.append(actor_loss)
    critic_loss_deque.append(critic_loss)

    NN_training_v1.update_priorities_of_buffer()

    if len(actor_loss_deque) == actor_loss_deque.maxlen:

        avg_actor_loss = sum(actor_loss_deque) / max_length
        avg_critic_loss = sum(critic_loss_deque) / max_length

        actor_loss_deque.clear()
        critic_loss_deque.clear()

        NN_training_v1.update_learning_rate(avg_actor_loss, avg_critic_loss)

        alpha, beta = NN_training_v1.calculate_alpha_beta(round + 1, rounds, alpha_max, alpha_min, beta_max, beta_min,factor)
        NN_training_v1.update_alpha_beta_of_buffer(alpha, beta)

        print("avg_actor_loss, avg_critic_loss:", avg_actor_loss,avg_critic_loss)
        print("alpha, beta", alpha, beta)