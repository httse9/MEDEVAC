from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle, save_pickle

if __name__ == "__main__":
    # load specfile
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters']= 40
    spec.optimization_hyperparams['alpha_theta']= 0.01
    spec.optimization_hyperparams['alpha_lamb']=0.03
    # Run Seldonian algorithm 
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test!")
        print("The solution found is:")
        print(solution)
        save_pickle("./solution0.01.pkl", solution)

        print("Primary objective evaluated on candidate dataset:")
        print(SA.evaluate_primary_objective(branch='candidate_selection', theta=solution))

        print("Primary objective evaluated on safety dataset:")
        print(SA.evaluate_primary_objective(branch='safety_test', theta=solution))
    else:
        print("No Solution Found")
