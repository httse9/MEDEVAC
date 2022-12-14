from seldonian.RL.Agents.Policies.Softmax import Softmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space, Continuous_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle
from seldonian.RL.environments.medevac import MedEvac
import autograd.numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_file", type=str, help="path to episodes file")
    parser.add_argument("--Z_n", type=int, default=12)

    args = parser.parse_args()
    episodes_file = args.episodes_file
    Z_n = args.Z_n

    print("Z_n:", Z_n, "episodes file:", episodes_file)

    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)

    # initialize policy
    env = MedEvac(Z_n = Z_n)
    num_features = env.num_features

    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["env"] = env #'medevac'
    hyperparam_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparam_and_setting_dict["basis"] = "Identity" 
    hyperparam_and_setting_dict["num_features"] = num_features

    env_description =  env.env_description
    policy = Softmax(hyperparam_and_setting_dict=hyperparam_and_setting_dict,
        env_description=env_description)
    env_kwargs={'gamma':1}
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
