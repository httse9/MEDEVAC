from seldonian.RL.environments.medevac import MedEvac
import numpy as np

def Myopic(env):
    _, zone = env.request

    if zone == -1:
        print('weird! This zone resquest sahould not have occured')
        action = env.M_n

    service_rate = env.mu_zm[zone, :]
    valid_actions = env.valid_actions.copy()

    best_m, best_t = env.M_n, -1
    for idx in range(env.M_n):
        if valid_actions[idx]:
            if service_rate[idx] > best_t:
                best_t = service_rate[idx]
                best_m = idx

    # if not valid_actions[best_m]:
    #     best_m = 4
    
    # print(valid_actions, best_m)
    return best_m


def Random(env):
    return np.random.choice(env.n_actions), 0.2

if __name__ == "__main__":
    Z_n = 12
    env = MedEvac(Z_n = Z_n)

    n_episodes = 10000

    returns = []
    for i in range(n_episodes):
        ret = 0
        env.reset()
        observation = env.get_observation()
        while not env.terminated():
            # action, _ = Random(env)
            action = Myopic(env)
            reward = env.transition(action)

            ret += reward
            observation = env.get_observation()

        returns.append(ret)

        if (i + 1) % 5000 == 0:
            print(i + 1)
    print(sum(returns) / len(returns))


# 34 zone
# performance (100000 episodes):
# Myopic: 1.713805617347962
# Random: 1.5133695351273548

# 12 zone
# performance (100000 episodes):
# Random: 1.4950880491323923
# Myopic: 1.6300647641017711

1.5191357808024584


# 34 zone invalid map to myopic:
# myopic: 2.6
# random: 2.0