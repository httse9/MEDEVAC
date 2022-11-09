from seldonian.utils.io_utils import load_pickle, save_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
from generate_data import state_to_number
from MEDEVAC import MEDEVAC
import numpy as np

def main():

    solution = load_pickle("./solution.pkl")
    print(solution)

    # plot_gradient_descent(solution, "Performance")

    env = MEDEVAC(speed=0)
    n_episodes = 1000
    gamma = 1

    returns = []
    for i in range(n_episodes):
        rewards = []
        done = False
        state = env.reset()
        while not done:
            q = solution[state_to_number(env, state)]
            # print(q.shape)
            valid_actions = state[-(env.M_n + 1):]
            # print(valid_actions.shape)

            q[valid_actions == 0] = float('-inf')

            action = np.argmax(q)
            next_state, r, done, _ = env.step(action)

            rewards.append(r)
            state = next_state

        ret = 0
        for r in reversed(rewards):
            ret = gamma * ret + r

        returns.append(ret)
    print(sum(returns) / len(returns))


if __name__ == "__main__":
    main()
