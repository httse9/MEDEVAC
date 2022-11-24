from seldonian.utils.io_utils import load_pickle, save_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.RL.environments.medevac import MedEvac
from seldonian.RL.RL_runner import (create_agent,run_trial_given_agent_and_env)
from seldonian.utils.stats_utils import weighted_sum_gamma
import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import re

Z_n = 34

def generate_episodes_and_cal_J():
    # Get trained model weights from running the Seldonian algo
    solution = load_pickle("./solution.pkl")

    # create env and agent
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = MedEvac(Z_n=Z_n)
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["basis"] = "Identity" 
    hyperparameter_and_setting_dict["num_features"] = 19
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

def main():

    solution = load_pickle("./solution.pkl")
    print(solution)

    log_files = os.listdir("./logs/")
    log_files = [int(re.sub("[^0-9]", "", f)) for f in log_files]

    cs_file = f'./logs/candidate_selection_log{max(log_files)}.p'
    solution_dict = load_pickle(cs_file)
    
    fig = plot_gradient_descent(solution_dict, primary_objective_name='J',\
        save=False)
    plt.show()

    # plot_gradient_descent(solution, "Performance")

    J = generate_episodes_and_cal_J()
    print(J)

    # env = MedEvac(Z_n = Z_n)
    # n_episodes = 1000
    # gamma = 1

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

    #         # valid_actions = observation[:env.n_actions]

    #         # q[valid_actions == 0] = float('-inf')

    #         # action = np.argmax(q)
    #         # print(action)
    #         reward = env.transition(action)

    #         ret += reward
    #         observation = env.get_observation()

    #     returns.append(ret)
    # print(sum(returns) / len(returns))


if __name__ == "__main__":
    main()
