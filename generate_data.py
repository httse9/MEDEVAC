from seldonian.dataset import Episode
from seldonian.utils.io_utils import save_pickle
from seldonian.RL.environments.medevac import MedEvac
import autograd.numpy as np

def Random(env):

    return np.random.choice(5), 0.2


def main():
    n_episodes = 50000
    Z_n = 12
    env = MedEvac(Z_n=Z_n)
    total_n_actions = 0
    n_actions_taken = np.zeros(env.n_actions)
    n_valid_actions_taken = np.zeros(env.n_actions)

    episodes = []
    for i in range(n_episodes):
        observations = []
        actions = []
        action_probs = []
        rewards = []

        env.reset()
        observation = env.get_observation()
        while not env.terminated():
            action, action_prob = Random(env)
            reward = env.transition(action)
            
            observations.append(observation)
            actions.append(action)
            action_probs.append(action_prob)
            rewards.append(reward)

            total_n_actions += 1
            n_actions_taken[action] += 1
            if env.valid_actions[action] == 1:
                n_valid_actions_taken[action] += 1

            observation = env.get_observation()

        episodes.append(Episode(observations, actions, rewards, action_probs))

    save_pickle(f"./MEDEVAC_{n_episodes}episodes.pkl", episodes)

    print(total_n_actions)
    print(n_actions_taken, n_valid_actions_taken, n_valid_actions_taken / n_actions_taken)

if __name__ == "__main__":
    main()
       

            



