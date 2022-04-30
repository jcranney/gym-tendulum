import gym
import numpy as np
from yaml import load
from agent import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800
    
    filename = 'cartpole.png'
    figure_file = "plots/" + filename
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
        
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print(f"episode: {i:d}, score: {score:0.1f}, average score: {avg_score:0.1f}, best score: {best_score:0.1f}")
    
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)