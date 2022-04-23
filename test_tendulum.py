import gym
import gym_tendulum

env = gym.make("tendulum-v0")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        observation, info = env.reset(return_info=True)

env.close()