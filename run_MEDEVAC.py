from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle, save_pickle

if __name__ == "__main__":
    # load specfile
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters']=30
    spec.optimization_hyperparams['alpha_theta']=0.1
    spec.optimization_hyperparams['alpha_lamb']=0.01
    # Run Seldonian algorithm 
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    if passed_safety:
        print("Passed safety test!")
        print("The solution found is:")
        print(solution)
        save_pickle("./solution.pkl", solution)
    else:
        print("No Solution Found")