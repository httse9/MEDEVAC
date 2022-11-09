from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle

def main():
    episodes_file = "./MEDEVAC_1000episodes.pkl"
    episodes = load_pickle(episodes_file)
    dataset = RLDataSet(episodes=episodes)

    # initialize policy
    state_space = Discrete_Space(0, 4826691)
    action_space = Discrete_Space(0, 4) # 0123: units, 4: no-op. Different from paper
    env_description =  Env_Description(state_space, action_space)
    policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
        env_description=env_description)
    env_kwargs={'gamma':1}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= 0']
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