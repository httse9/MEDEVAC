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

Z_n = 12

def generate_episodes_and_cal_J():
    # Get trained model weights from running the Seldonian algo
    solution = load_pickle("./solution.pkl")

    # create env and agent
    env = MedEvac(Z_n=Z_n)
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = env
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["basis"] = "Identity" 
    hyperparameter_and_setting_dict["num_features"] = env.num_features
    agent = create_agent(hyperparameter_and_setting_dict)
    env = hyperparameter_and_setting_dict["env"]

    # set agent's weights to the trained model weights
    agent.set_new_params(solution)

    # generate episodes
    num_episodes = 1000
    episodes = run_trial_given_agent_and_env(
        agent=agent,env=env,num_episodes=num_episodes)

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
    J = np.mean(returns)
    return J

def main(file):
    pass
    # solution = load_pickle("./solution.pkl")
    # print(solution)

    # # observe probabilities
    # env = MedEvac(Z_n = Z_n)
    # n_episodes = 1000
    # gamma = 1
    # n_actions_taken = np.zeros(env.n_actions)
    # n_valid_actions_taken = np.zeros(env.n_actions)

    # returns = []
    # for i in range(n_episodes):
    #     ret = 0

    #     env.reset()
    #     observation = env.get_observation()
    #     while not env.terminated():
    #         q = observation @ solution
    #         action_prob = np.exp(q - np.max(q))
    #         action_prob /= action_prob.sum()
    #         action = np.random.choice(5, p=action_prob)
    #         print(action_prob, action, np.argmax(q))

    #         n_actions_taken[action] += 1
    #         if env.valid_actions[action] == 1:
    #             n_valid_actions_taken[action] += 1

    #         # mask out invalid actions, which is not required anymore
    #         # valid_actions = observation[:env.n_actions]
    #         # q[valid_actions == 0] = float('-inf')

    #         reward = env.transition(action)

    #         ret += reward
    #         observation = env.get_observation()

    #     returns.append(ret)
    # print(sum(returns) / len(returns))
    # print(n_actions_taken, n_valid_actions_taken, n_valid_actions_taken / n_actions_taken)

    # # calculate performance
    # J = generate_episodes_and_cal_J()
    # print(J)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str, help="name of log file")
    parser.add_argument("--save", action="store_true", help="whether to save gradient descent plot")
    parser.add_argument("--savename", help="name of saved gd plot", type=str, \
        default=None)

    args = parser.parse_args()
    file = args.file

    if file is None:
        log_files = os.listdir("./logs/")
        log_files = [int(re.sub("[^0-9]", "", f)) for f in log_files]

        file = './logs/candidate_selection_log' + str(max(log_files)) + '.p'
    solution_dict = load_pickle(file)

    # plot gradient descent
    savename = args.savename
    if args.save and savename is None:
        savename = ".".join(file.split(".")[:-1] + ["png"])
    fig = plot_gradient_descent(solution_dict, primary_objective_name='J',\
        save=args.save, savename=savename)
    plt.show()
