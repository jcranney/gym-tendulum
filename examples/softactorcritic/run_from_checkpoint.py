import argparse
import datetime
import gym
import gym_tendulum
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from pygame_recorder import ScreenRecorder

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="tendulum-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--checkpoint_file', default=None, help='path to checkpoint file')
args = parser.parse_args()

if args.checkpoint_file is None:
    # if not defined, grab the latest
    checkpoint_file = "checkpoints\\sac_checkpoint_tendulum-v0__quadcost_ep_0010"
    print("WARNING: CHECKPOINT NOT DEFINED, USING\n"+checkpoint_file)
else:
    checkpoint_file = args.checkpoint_file
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_checkpoint(checkpoint_file)
episode = int(checkpoint_file[-4:])

import pygame
clock = pygame.time.Clock()
done = False
state = env.reset()
env.render()
recorder = ScreenRecorder(env.screen.get_width(), env.screen.get_height(), 60, out_file=f"output_{episode:04d}.avi")
episode_reward = 0.0
while not done:
    env.render(render_text=f"Ep: {episode:d}")
    recorder.capture_frame(env.screen)
    action = agent.select_action(state)  # Sample action from policy
    state, reward, done, _ = env.step(action) # Step
    episode_reward += reward
    clock.tick(60)

recorder.end_recording()
env.close()