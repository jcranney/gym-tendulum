import gym
import gym_tendulum

env = gym.make("tendulum-v0")
observation, info = env.reset(seed=42, return_info=True)

done = False
while not done:
    action = -0.1
    observation, reward, done, info = env.step(action)
    env.render()

env.close()