from seldonian.utils.io_utils import load_pickle, save_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
from generate_data import state_to_number
from seldonian.RL.environments.medevac import MedEvac
import numpy as np

def main():

    solution = load_pickle("./solution.pkl")
    print(solution)

    # plot_gradient_descent(solution, "Performance")

    env = MedEvac()
    n_episodes = 1000
    gamma = 1

    returns = []
    for i in range(n_episodes):
        ret = 0

        env.reset()
        observation = env.get_observation()
        while not env.terminated():
            q = observation @ solution
            print(q)

            valid_actions = observation[:env.n_actions]

            q[valid_actions == 0] = float('-inf')

            action = np.argmax(q)
            reward = env.transition(action)

            ret += reward
            observation = env.get_observation()

        returns.append(ret)
    print(sum(returns) / len(returns))


if __name__ == "__main__":
    main()
