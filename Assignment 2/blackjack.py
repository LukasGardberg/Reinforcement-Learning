###########################################################

import gym
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import copy
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import random

# Solver for OpenAi's Blackjack environment using Q-learning

# Initialize environment

env = gym.make('Blackjack-v0')
env.reset()

n_actions = env.action_space.n

# input: a deterministic policy to be evaluated, taken from the book's example


def create_initial_policy():
    # initialize policy with 200 states
    # player's sum 12-21 * dealer showing 1-10 * usable ace
    # set all actions to hit=1 ...

    policy = np.ones([10, 10, 2])

    # ... except when player has 20 or 21, then it is stick
    policy[8:, :, :] = 0

    return policy


current_policy = create_initial_policy()

# env.step(action) returns: ((playersum, dealercard, usable_ace), reward, done, info)

# create object for state action-values Q:

q_table = defaultdict(float)

q_table['terminal', 0] = 0.0

q_table['terminal', 1] = 0.0

# set gamma to 1.0 as it is en episodic environment

gamma = 1.0

epsilon = 0.8  # 0.2

alpha = 0.5

min_eps = 0.01

decay = .99995

###################################

# State space dimensions
state_space = (10, 10, 2)


def epsilon_greedy(policy, state, epsilon=0.8):
    # Returns an action according to an epsilon greedy policy derived from policy

    # player sum index = player sum - 12
    # Convert boolean to index
    state_index = (state[0] - 12, state[1] - 1, int(state[2] == True))

    if random.random() < epsilon:
        # Return random action
        return random.randint(0, n_actions - 1)
    else:
        # Return action according to policy
        return int(policy[state_index])


def update_policy(q_table):
    # Return a greedy policy derived from q_table
    # Could be improved: only iterate through states where Q has been updated?
    # Currently iterates through all.

    # Initialize policy
    policy = np.zeros([10, 10, 2])

    # Loop over all states
    for player_sum_index in range(state_space[0]):
        for dealer_card_index in range(state_space[1]):
            for usable_ace_index in range(state_space[2]):
                # Optimal action in each state s: a = argmax(Q(s,a))

                # q_table is indexed by (player sum, dealer card, usable ace) = (12-21, 1-10, True/False)
                action_values = [q_table[(player_sum_index + 12, dealer_card_index + 1, usable_ace_index == 1),
                                         action] for action in range(n_actions)]

                policy[player_sum_index, dealer_card_index, usable_ace_index] = np.argmax(action_values)

    return policy


def q_control(q_table, current_policy, gamma=1.0, epsilon=0.8, alpha=0.5):
    # gamma: discount rate
    # epsilon: exploration rate
    # alpha: learning rate / step size

    # Implements the q learning algorithm

    # Initialize state (start new game)
    observation = env.reset()     # (player_sum, dealer_sum, usable_ace)
    done = False

    # Loop over episode steps
    while not done:
        # Get next action using current Q(s,a), epsilon greedy
        action = epsilon_greedy(current_policy, observation, epsilon)

        # Take action
        new_observation, reward, done, _ = env.step(action)

        # Calculate the maximum value from all possible actions in current state
        max_next_value = np.amax([q_table[new_observation, temp_act] for temp_act in range(n_actions)])

        # Calculate difference needed in update rule
        temporal_difference = reward + gamma * max_next_value - q_table[observation, action]

        # Update Q(s,a)
        q_table[observation, action] = q_table[observation, action] + alpha * temporal_difference

        observation = new_observation  # Update observation

        current_policy = update_policy(q_table)  # Update policy

    # Episode has ended

    # Return updated q_table and optimal policy
    return q_table, current_policy

# loop over episodes


for i in tqdm(range(200000)):

    # Decay epsilon

    if (epsilon > min_eps) and (i % 1000 == 0):
        epsilon -= (epsilon * decay)

    q_table, current_policy = q_control(q_table, current_policy, gamma=gamma, epsilon=epsilon, alpha=alpha)

###########################################################

print(i + 1, "episodes processed")

###################################


# print the policy

q_list_ace = []

q_list_noace = []


def get_value_function(q_table, policy):
    # Calculates V(s) = Q(s, pi(s))

    value_func = np.zeros(state_space)

    for player_sum_index in range(state_space[0]):
        for dealer_card_index in range(state_space[1]):

            value_func[(player_sum_index, dealer_card_index, 1)] = \
                q_table[(player_sum_index + 12, dealer_card_index + 1, True), policy[player_sum_index, dealer_card_index, 1]]

            value_func[(player_sum_index, dealer_card_index, 0)] = \
                q_table[(player_sum_index + 12, dealer_card_index + 1, False), policy[player_sum_index, dealer_card_index, 0]]

    return value_func


value_function = get_value_function(q_table, current_policy)

# Plot value function
plt.figure(figsize=(8, 8))

sns.heatmap(np.reshape(value_function[:, :, 0], (10, 10)), cmap="YlGnBu", annot=True,
            xticklabels=range(1, 11, 1), yticklabels=range(12, 22, 1),
            cbar=False, square=True)

plt.show()

# Plot policy
for playersum in range(10):

    for dealercard in range(10):
        q_list_ace.append(current_policy[(playersum, dealercard, 1)])

        q_list_noace.append(current_policy[(playersum, dealercard, 0)])


plt.figure(figsize=(4, 4))

sns.heatmap(np.reshape(q_list_ace, (10, 10)), cmap="YlGnBu", annot=True,
            xticklabels=range(1, 11, 1), yticklabels=range(12, 22, 1),
            cbar=False, square=True)

plt.show()

plt.figure(figsize=(4, 4))

sns.heatmap(np.reshape(q_list_noace, (10, 10)), cmap="YlGnBu", annot=True,
            xticklabels=range(1, 11, 1), yticklabels=range(12, 22, 1),
            cbar=False, square=True)

plt.show()

