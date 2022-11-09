from seldonian.dataset import Episode
from seldonian.utils.io_utils import save_pickle
from MEDEVAC import MEDEVAC, Random
import numpy as np

def state_to_number(env, state):
    state = state[: - (env.M_n + 1)].reshape(-1, env.Z_n)
    assert state.shape == (env.M_n + env.K_n, env.Z_n)

    # MEDEVAC Unit Status
    state_numbers = []
    for m in range(env.M_n):
        if state[m].sum() == 0: # unit m not dispatched
            state_numbers.append(0)
        else: # find where unit m is dispatched
            state_numbers.append(np.argmax(state[m]) + 1)

    # print(state[:env.M_n])
    # print(state_numbers)

    # Request Status
    for k in range(env.K_n):
        for z in range(env.Z_n):
            if state[env.M_n + k, z] == 1:
                state_numbers += [k + 1, z + 1]
    
    if len(state_numbers) < env.M_n + 2:
        state_numbers += [0, 0]

    # print(state[env.M_n:])
    # print(state_numbers)
    # print("************")

    # construct number
    assert len(state_numbers) == env.M_n + 2
    number = 0
    for i in range(env.M_n + 2):
        number = (env.Z_n + 1) * number + state_numbers[i]

    # print(number)
    return number



def main():
    n_episodes = 1000
    env = MEDEVAC(speed=0, debug=True)

    episodes = []
    for i in range(n_episodes):
        states = []
        actions = []
        action_probs = []
        rewards = []
        done = False
        state = env.reset()
        while not done:
            action, action_prob = Random(env, state)
            next_state, r, done, _ = env.step(action)

            states.append(state_to_number(env, state))
            actions.append(action)
            action_probs.append(action_prob)
            rewards.append(r)

            state = next_state

        episodes.append(Episode(states, actions, rewards, action_probs))

    save_pickle(f"./MEDEVAC_{n_episodes}episodes.pkl", episodes)

if __name__ == "__main__":
    main()
       

            



