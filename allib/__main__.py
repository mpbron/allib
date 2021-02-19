from allib import app
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="allib",
        description="Active Learning Library (allib) - Benchmarking tool"
    )
    parser.add_argument("dataset", help="The path to the dataset")
    parser.add_argument("target", help="The target of the results")
    parser.add_argument("al_choice", help="The choice for the Active Learnign method")
    parser.add_argument("fe_choice", help="The choice for the Feature Extraction method")
    args = parser.parse_args()

    app.run_benchmark(args.dataset, 
                      args.target, 
                      args.al_choice, 
                      args.fe_choice)