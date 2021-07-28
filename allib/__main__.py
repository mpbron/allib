from allib import app
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="allib",
        description="Active Learning Library (allib) - Benchmarking tool"
    )
    parser.add_argument("dataset", help="The path to the dataset")
    parser.add_argument("target", help="The target of the results")
    parser.add_argument("al_choice", help="The choice for the Active Learning method")
    parser.add_argument("fe_choice", help="The choice for the Feature Extraction method")
    parser.add_argument("est_choice", help="The choice for the Estimation method")
    parser.add_argument("stop_choice", help="The choice for the Stopping Criterion")
    args = parser.parse_args()

    app.run_benchmark(args.dataset, 
                      args.target, 
                      args.al_choice, 
                      args.fe_choice,
                      args.est_choice,
                      args.stop_choice)