import sys
import numpy as np
import matplotlib.pyplot as plt
from dtree import DNode
from qtree import QTree
from qtree_agent import QLearningAgent
import gym

import logging
import argparse
import json

def main(args):
    # Create an environment
    envname = 'Acrobot-v1'
    env = gym.make(envname)
    SEED = 505
    env.seed(SEED)

    # Explore state (observation) space
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

    # Explore action space
    print("Action space:", env.action_space)

    n_bins = 2
    tq = QTree(env.observation_space.low, env.observation_space.high, n_bins, env.action_space.n, adaptive=args.adaptive, p=args.p)
    agent = QLearningAgent(env, tq)

    def run(agent, env, num_episodes=10000, mode='train'):
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        max_avg_score = -np.inf
        for i_episode in range(1, num_episodes+1):
            # Initialize episode
            state = env.reset()
            action = agent.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)

            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                if i_episode % 100 == 0:
                    print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                    sys.stdout.flush()
        return scores

    scores = run(agent, env, num_episodes=args.num_episodes)

    with open('score_history.json', 'w') as f:
        json.dump({'scores': scores, 'args': args, 'n_bins': n_bins, 'seed': SEED, 'envname': envname}, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loglevel', type=str, default='DEBUG')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('-n', '--num-episodes', type=int, default=10000)
    parser.add_argument('--p', type=int, default=1000)

    args = parser.parse_args()
    main(args)
