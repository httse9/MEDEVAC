from seldonian.utils.io_utils import load_pickle, save_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.RL.environments.medevac import MedEvac
from seldonian.RL.RL_runner import (create_agent,run_trial_given_agent_and_env)
from seldonian.utils.stats_utils import weighted_sum_gamma
import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse

def performance_on_train_set(sol):

    mean_ret = []
    for ep in episodes:
        ret = ep.rewards.sum()
        mean_ret.append(ret)

    print("Random Policy Mean Return on Training Set:", np.mean(mean_ret))

    n_candidate = int(round(n_episodes*0.4))

    eps = episodes[:n_candidate]
    mean_ret2 = []
    for ep in eps:
        ret = IS_expected_return(sol, ep)
        mean_ret2.append(ret)

    print("Learnt Policy Mean Return on Training Candidate Set:", np.mean(mean_ret2))

    eps = episodes[n_candidate:]
    mean_ret2 = []
    for ep in eps:
        ret = IS_expected_return(sol, ep)
        mean_ret2.append(ret)

    print("Learnt Policy Mean Return on Training Safety Set:", np.mean(mean_ret2))

def performance_on_new_set(solution):
    env = MedEvac(Z_n = Z_n)

    returns = []
    for i in range(50000):
        ret = 0

        env.reset()
        observation = env.get_observation()
        while not env.terminated():
            q = observation @ solution
            action_prob = np.exp(q - np.max(q))
            action_prob /= action_prob.sum()
            action = np.random.choice(5, p=action_prob)
            # print(action_prob, action, np.argmax(q))

            reward = env.transition(action)

            ret += reward
            observation = env.get_observation()

        returns.append(ret)

    print("Learnt Policy Mean Return on New Set:", np.mean(returns))

# def new_set_state_overlap(solution):
#     env = MedEvac(Z_n = Z_n)
#     returns = []
#     for i in range(1000):
#         ret = 0

#         env.reset()
#         observation = env.get_observation()
#         while not env.terminated():
#             q = observation @ solution
#             action_prob = np.exp(q - np.max(q))
#             action_prob /= action_prob.sum()
#             action = np.random.choice(5, p=action_prob)
#             # print(action_prob, action, np.argmax(q))

#             reward = env.transition(action)

#             ret += reward
#             observation = env.get_observation()

#         returns.append(ret)

def IS_expected_return(sol, ep):
    ret = ep.rewards.sum()
    # print(ret)

    frac = 1.0
    for ob, action in zip(ep.observations, ep.actions):
        # print(ob.shape)
        prob = ob @ sol
        # print(sol.shape)
        prob = np.exp(prob - np.max(prob))
        prob /= prob.sum()

        f = prob[action] / 0.2
        frac *= f
        # print(f)
    # print(frac)
    return frac * ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="name of solution file", type=str)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--Z_n", type=int, default=12)

    args = parser.parse_args()

    Z_n = args.Z_n
    n_episodes = args.n_episodes
    file = args.file

    print("Z_n:", Z_n, "n_episodes:", n_episodes, "file:", file)

    episodes_file = f"/media/htse/MEDEVAC_{n_episodes}episodes.pkl"
    episodes = load_pickle(episodes_file)

    solution = load_pickle(file)

    performance_on_train_set(solution)
    performance_on_new_set(solution)
