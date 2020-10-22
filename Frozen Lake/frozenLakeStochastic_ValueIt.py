import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Solution to the stochastic frozen lake RL problem using Value Iteration.

gym.envs.register(
    id='FrozenLakeSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True},
    max_episode_steps=100,
)

env = gym.make('FrozenLakeSlippery-v0')

# reset the environment, get an observation
# obs represents the initial state the agent is in
obs = env.reset()

_ = env.render()  # Shows the current environment (console print)


def value_iteration(env, threshold=0.001):

    V = np.zeros(env.nS)

    n_actions = env.action_space.n
    # optimal_actions = np.zeros(env.nS)  # Matrix to store optimal action for each state

    while True:

        delta = 0

        V_new = np.zeros(env.nS)

        for state in range(env.nS):

            # Initialize array of values of all potential actions in current state
            # Size = number of actions available
            potential_values = np.zeros(n_actions)
            # V = V_new

            for action in range(n_actions):
                # Get next state, reward etc from environment after action is chosen

                for prob_next_state, next_state, reward, done in env.P[state][action]:
                    potential_values[action] += prob_next_state * (reward + V[next_state])

            V_new[state] = max(potential_values)

            # Update V[state] = V_new[state] here?

            delta = max(delta, np.abs(V[state] - V_new[state]))

        V = V_new  # Update value function

        if delta < threshold:
            break

    return V


def create_optimal_policy(env, v_star):
    # Outputs a policy (list) of optimal actions where
    # optimal_policy[state] is the optimal action in that state.
    #
    # Here 'env' is the deterministic environment in order to easily
    # obtain the next state resulting from an action.

    optimal_policy = np.zeros(env.nS)

    for state in range(env.nS):

        # List of possible next states, indexed by action number
        next_states = np.zeros(env.action_space.n)

        # For all possible actions, get which state the action results in
        for action in range(env.action_space.n):
            _, next_state, _, _ = env.P[state][action][0]
            next_states[action] = next_state

        # Let the policy be the action in each state that results in the highest value
        optimal_policy[state] = np.argmax([v_star[np.int(s)] for s in next_states])

    return optimal_policy

# Using the deterministic environment to get next state after action
gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
)

t1 = time.time()

value_func = value_iteration(env)

env_determ = gym.make('FrozenLakeNotSlippery-v0')

opt_acts = create_optimal_policy(env_determ, value_func)

print(f'Time: {time.time() - t1}')

print(value_func.reshape(4, 4))
print(opt_acts.reshape(4, 4))

env.reset()
env.render()
plt.figure(figsize=(4, 4))
#plt.imshow(value_func.reshape(4, 4), interpolation='nearest')
sns.heatmap(value_func.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False)
plt.show()
