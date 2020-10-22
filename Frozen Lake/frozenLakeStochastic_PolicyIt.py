import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Solution to the stochastic frozen lake RL problem using Policy Iteration.

# Set up environment
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


# Creates a policy which chooses an action randomly
def random_policy_with_prob(n_states, n_actions):
    policy = np.ones([n_states, n_actions]) / n_actions
    return policy


# Evaluates a certain policy and returns a value function V(s)
def evaluate_policy(env, policy, threshold=0.001):

    # policy: list, size: (n_states, n_actions) (16, 4)
    # contains the prob. for each action in each state.

    V = np.zeros(env.nS)  # Value function

    # Evaluate policy until we reach a certain threshold
    while True:

        delta = 0  # initialize max difference in values

        V_new = np.zeros(env.nS)  # Initialize new value function

        # Evaluate all states
        for state in range(env.nS):

            # Evaluate all actions
            for action, action_prob in enumerate(policy[state]):
                # Get next state, reward etc from environment after action is chosen

                for prob_next_state, next_state, reward, done in env.P[state][action]:
                    # Sum up according to update rule
                    V_new[state] += action_prob * prob_next_state * (reward + V[next_state])

            # Obtain maximum difference compared to old value function
            delta = max(delta, np.abs(V[state] - V_new[state]))

        V = V_new  # Update value function

        # If we have reached our threshold, we are done
        if delta < threshold:
            break

    return V


# Policy iteration


def policy_iteration(env):

    # Initialize V and PI (arbitrarily)

    policy = random_policy_with_prob(16, 4)

    # Policy improvement

    n_actions = env.action_space.n

    # Loop until we obtain a stable policy
    while True:

        stable_policy = True
        value_func = evaluate_policy(env, policy)

        for state in range(env.nS):

            old_action = np.array(policy[state])  # list on probabilities of actions in a state, ex [0, 1, 0, 0].

            potential_values = np.zeros(n_actions)  # Initialize list of values to base policy on

            # Evaluate for all actions
            for action in range(n_actions):
                # Get next state, reward etc from environment after action is chosen

                for prob_next_state, next_state, reward, done in env.P[state][action]:
                    # Calculate sum and store
                    potential_values[action] += prob_next_state * (reward + value_func[next_state])

            # Update PI
            policy[state] = np.zeros(n_actions)  # All other actions 0 probability
            policy[state][np.argmax(potential_values)] = 1  # Chosen action prob. of 1

            # If "optimal" actions differ from old actions, the policy is not stable
            if not np.array_equal(old_action, policy[state]):
                stable_policy = False

        # No difference in actions, we are done
        if stable_policy:
            break
        else:  # Reevaluate policy
            continue

    # Convert policy matrix to list of optimal actions
    policy = [np.argmax(actions) for actions in policy]

    return value_func, np.asarray(policy)

# Perform the policy iteration

t1 = time.time()

value_func, policy = policy_iteration(env)

print(f'Time: {time.time() - t1}')

print(value_func.reshape(4, 4))
print(policy.reshape(4, 4))

# plot

env.reset()
env.render()
plt.figure(figsize=(6, 6))
#plt.imshow(value_func.reshape(4, 4), interpolation='nearest')
sns.heatmap(value_func.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False)
plt.show()
