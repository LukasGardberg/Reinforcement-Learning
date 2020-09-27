from tqdm import tqdm
import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load frozen lake env and change to deterministic behavior
gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
)

# Create an environment to interact with
env = gym.make('FrozenLakeNotSlippery-v0')

# reset the environment, get an observation
# obs represents the initial state the agent is in
obs = env.reset()

# Prints the number of actions and states
print(env.action_space)
print(env.observation_space)

#env.render('rgb_array')
_ = env.render()  # Shows the current environment (console print)

random_action = env.action_space.sample()  # Samples an action
print(random_action)

current_state = env.reset()
done = False

# Runs one timestep of the environments dynamics
current_state, reward, done, info = env.step(env.action_space.sample())
print('state: ', current_state)
print('reward: ', reward)
print('done: ', done)
print('info: ', info)

# Step through the environment with random sampled actions

current_state = env.reset()
done = False
while not done:
    current_state, reward, done, info = env.step(env.action_space.sample())   
    env.render()

# Create own agent, and step through environment


class create_agent():
    # create a grid environment
    
    def __init__(self, name='random', actions=gym.spaces.Discrete(4)):
        self.name = name.lower()
        self.actions = actions
        
        if 'random' in name:
            self.policy = 'random'
        else:
            self.policy = 'random'

    def act(self):
        return self.actions.sample()


current_state = env.reset()
done = False

agent = create_agent('random')

while not done:
    current_state, reward, done, info = env.step(agent.act())

env.render()  # Final environment


'''

How to implement policy evaluation?

'''


# In[ ]:


# no need to reassign states to env and execute step()

# state, action
print('here')
print(env.P[0][0])
# probability, state', reward, done


def random_policy_with_prob(n_states, n_actions):
    policy = np.ones([n_states, n_actions]) / n_actions
    return policy


def evaluate_policy(env, policy, threshold=0.001):

    V = np.zeros(env.nS)

    while True:

        delta = 0

        V_new = np.zeros(env.nS)

        for state in range(env.nS):

            for action, action_prob in enumerate(policy[state]):
                # Get next state, reward etc from environment after action is chosen
                (_, next_state, reward, done) = env.P[state][action][0]
                V_new[state] += action_prob * (reward + V[next_state])

            delta = max(delta, np.abs(V[state] - V_new[state]))

        V = V_new  # Update value function

        if delta < threshold:
            break

    return V


random_policy = random_policy_with_prob(16, 4)

value_func = evaluate_policy(env, random_policy)

# Done, plot

env.reset()
env.render()
plt.figure(figsize=(6, 6))
#plt.imshow(value_func.reshape(4, 4), interpolation='nearest')
sns.heatmap(value_func.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False)
plt.show()


