from seldonian.RL.Agents.Policies.Softmax import Softmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space, Continuous_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle
from seldonian.RL.environments.medevac import MedEvac
import autograd.numpy as np

def main():
    n_episodes = 50000
    episodes_file = f"./MEDEVAC_{n_episodes}episodes.pkl"
    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)

    # initialize policy
    Z_n = 12
    env = MedEvac(Z_n = Z_n)
    num_features = env.num_features

    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["env"] = env
    hyperparam_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparam_and_setting_dict["basis"] = "Identity" 
    hyperparam_and_setting_dict["num_features"] = num_features

    env_description =  env.env_description
    policy = Softmax(hyperparam_and_setting_dict=hyperparam_and_setting_dict,
        env_description=env_description)
    env_kwargs={'gamma':1.0}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= 0']
    deltas= [0.05]

    spec = createRLSpec(
        dataset=dataset,
        policy=policy,
        constraint_strs=constraint_strs,
        deltas=deltas,
        env_kwargs=env_kwargs,
        save=True,
        save_dir='.',
        verbose=True)

if __name__ == "__main__":
    main()