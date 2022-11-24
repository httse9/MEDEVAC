from seldonian.dataset import Episode
from seldonian.utils.io_utils import save_pickle
from seldonian.RL.environments.medevac import MedEvac
import autograd.numpy as np

def Random(env):
    # valid = env.valid_actions.copy()
    # probs = valid/ np.sum(valid)
    # action = np.random.choice(env.n_actions, p=probs)

    return np.random.choice(5), 0.2

    return action, probs[action]


def main():
    n_episodes = 1000
    Z_n = 34
    env = MedEvac(Z_n=Z_n)

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

            observation = env.get_observation()

        episodes.append(Episode(observations, actions, rewards, action_probs))

    save_pickle(f"./MEDEVAC_{n_episodes}episodes.pkl", episodes)

if __name__ == "__main__":
    main()
       

            



