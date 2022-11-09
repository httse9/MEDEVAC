import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import (create_env,
    create_agent,run_trial_given_agent_and_env)

if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    performance_metric = 'J(pi_new)'
    n_trials = 20
    data_fracs = np.logspace(-2.3,0,10)
    n_workers = 8
    verbose=True
    results_dir = f'results/MEDEVAC_{n_trials}trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'MEDEVAC_{n_trials}trials.png')
    n_episodes_for_eval = 1000

    # Load spec
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.01
    spec.optimization_hyperparams['alpha_lamb'] = 0.01

    