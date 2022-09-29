from pathlib import Path
from allib import app
import argparse
from multiprocessing import Pool

from allib.configurations.catalog import ExperimentCombination

from .module.catalog import ActiveLearningTasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="allib",
        description="Active Learning Library (allib) - Benchmarking tool"
    )
    parser.add_argument("-d", "--dataset", help="The path to the dataset", type=Path)
    parser.add_argument("-t", "--target", help="The target of the results", type=Path)
    parser.add_argument("-e", "--exp_choice", help="The experiment method", type=ExperimentCombination)
    parser.add_argument("--problem", "--task", metavar="problem", default="TAR", type=ActiveLearningTasks, help="The problem specification")
    parser.add_argument("-p", "--pos_label", metavar="POS", default="Relevant", help="The label that denotes the positive class")
    parser.add_argument("-n", "--neg_label", metavar="NEG", default="Irrelevant", help="The label that denotes the negative class")
    parser.add_argument("-q", "--nthreads", default=4, type=int)
    parser.add_argument("-m", "--samplesize", default=15, type=int)
    args = parser.parse_args()
    pool = Pool(args.nthreads)
    if args.problem == ActiveLearningTasks.TAR:
        app.tar_benchmark(args.dataset, 
                          args.target, 
                          args.exp_choice, 
                          args.pos_label,
                          args.neg_label)
    else:
        raise NotImplementedError("Other tasks have no Benchmark procedure yet")