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
import pickle

from timeit import default_timer

def main(args):
    # Create an environment
    envname = args.envname
    env = gym.make(envname)
    SEED = 505
    env.seed(SEED)

    # Explore state (observation) space
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

    # Explore action space
    print("Action space:", env.action_space)

    n_bins = args.n_bins
    if args.p is None:
        ARGSP = np.inf
    else:
        ARGSP = args.p

    tq = QTree(env.observation_space.low, 
        env.observation_space.high, 
        n_bins, 
        env.action_space.n, 
        adaptive=args.adaptive, 
        p=ARGSP)
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
            timer_start = default_timer()
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)
            timer_end = default_timer()
            timer_diff = timer_end - timer_start
            timer_str = f"{timer_diff:.04f} sec/episode"
            #print(timer_str, end="\r")
            # Save final score
            scores.append(total_reward)

            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                if i_episode % 100 == 0:
                    print("\nEpisode {}/{} | Max Average Score: {}, tree-size: {}".format(i_episode, num_episodes, max_avg_score, len(tq.qtree)), end="\r")
                    sys.stdout.flush()

                    #if i_episode % 300 == 0:
                    #    print('----- DEBUG ENTRY -----')
        return scores

    scores = run(agent, env, num_episodes=args.num_episodes)

    with open(args.score_file, 'w') as f:
        json.dump({'scores': scores, 'args': vars(args), 'n_bins': n_bins, 'seed': SEED, 'envname': envname}, f)

    with open(args.score_file+'_qtree.pickle', 'wb') as f:
        pickle.dump(tq.qtree, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loglevel', type=str, default='DEBUG')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('-n', '--num-episodes', type=int, default=10000)
    parser.add_argument('--p', type=int)
    parser.add_argument('--score-file', type=str, default='score_history.json')
    parser.add_argument('--envname', type=str)
    parser.add_argument('--n-bins', type=int, default=2)

    args = parser.parse_args()
    main(args)
