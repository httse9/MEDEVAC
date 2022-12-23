from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle, save_pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sol_name", help="name of solution file", type=str)
    parser.add_argument("--spec_name", help="name of generated spec file", type=str, default="spec.pkl")
    parser.add_argument("--num_iters", help="number of iterations to run", type=int, default=40)
    parser.add_argument("--alpha_theta", help="learning rate for primal variables", type=float, default=0.003)
    parser.add_argument("--alpha_lamb", help="learning rate for dual variable", type=float, default=0.03)

    args = parser.parse_args()

    # load specfile
    specfile = args.spec_name
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = args.num_iters
    spec.optimization_hyperparams['alpha_theta'] = args.alpha_theta
    spec.optimization_hyperparams['alpha_lamb'] = args.alpha_lamb

    print("***********************")
    print("Specfile:", specfile)
    print("num_iters:", args.num_iters)
    print("alpha_theta:", args.alpha_theta)
    print("alpha_lamd:", args.alpha_lamb)
    print("Solution save file name:", args.sol_name)
    print("***********************")

    # Run Seldonian algorithm 
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test!")
        # print("The solution found is:")
        # print(solution)
        save_pickle(args.sol_name , solution)

        print("Primary objective evaluated on candidate dataset:")
        print(SA.evaluate_primary_objective(branch='candidate_selection', theta=solution))

        print("Primary objective evaluated on safety dataset:")
        print(SA.evaluate_primary_objective(branch='safety_test', theta=solution))
    else:
        print("No Solution Found")
