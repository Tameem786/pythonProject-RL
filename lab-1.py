import gymnasium as gym

env = gym.make('Acrobot-v1')

print(env.observation_space)
print(env.action_space)

env.close()