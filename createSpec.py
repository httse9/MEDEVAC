from seldonian.RL.Agents.Policies.Softmax import Softmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space, Continuous_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle
from seldonian.RL.environments.medevac import MedEvac
import autograd.numpy as np

def main():
    episodes_file = "./MEDEVAC_1000episodes.pkl"
    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)

    # initialize policy
    num_features = 19
    Z_n = 34
    each_dim_bound = np.array([[0.0, 1.0]])
    observation_space_bounds = np.repeat(each_dim_bound, num_features, axis=0)
    # observation_space_bounds = np.array([
    #     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    #     [0, 20], [0, 200], [0, 200], [0, 200],
    #     [0, 20], [0, 200], [0, 200], [0, 200],
    #     [0, 20], [0, 200], [0, 200], [0, 200], 
    #     [0, 1], [1, 1]
    # ])
    observation_space = Continuous_Space(observation_space_bounds)

    action_space = Discrete_Space(0, 4) # 0123: units, 4: no-op. Different from paper

    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["env"] = MedEvac(Z_n=Z_n)
    hyperparam_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparam_and_setting_dict["basis"] = "Identity" 
    hyperparam_and_setting_dict["num_features"] = num_features

    env_description =  Env_Description(observation_space, action_space)
    policy = Softmax(hyperparam_and_setting_dict=hyperparam_and_setting_dict,
        env_description=env_description)
    env_kwargs={'gamma':1.0}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= 1']
    deltas=[0.05]

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