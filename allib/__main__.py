from pathlib import Path
from allib import app
import argparse
from allib.benchmarking.datasets import DatasetType, TarDataset

from allib.configurations.catalog import ExperimentCombination
from allib.benchmarking.jobs import create_jobs

from .module.catalog import ActiveLearningTasks, ApplicationModules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="allib", description="Active Learning Library (allib) - Benchmarking tool"
    )
    parser.add_argument("module", type=ApplicationModules, default=ApplicationModules.BENCHMARK)
    parser.add_argument("-j", "--jobpath", help="Path of jobfiles", type=Path)
    parser.add_argument("-r", "--seed", type=int, default=None)
    parser.add_argument("-m", "--datasettype", type=DatasetType, help="Dataset Type")
    parser.add_argument("-v", "--verboseplotting", type=bool, default=False)
    parser.add_argument("-s", "--stopinterval", type=int, default=None)
    parser.add_argument("-d", "--dataset", help="The path to the dataset", type=Path)
    parser.add_argument("-t", "--target", help="The target of the results", type=Path)
    parser.add_argument("--highcpu", type=int,default=8)
    parser.add_argument("--lowcpu", type=int,default=2)
    parser.add_argument("--mediumcpu", type=int,default=4)
    parser.add_argument(
        "-e", "--exp_choice", help="The experiment method", type=ExperimentCombination
    )
    parser.add_argument(
        "--problem",
        "--task",
        metavar="problem",
        default="TAR",
        type=ActiveLearningTasks,
        help="The problem specification",
    )
    parser.add_argument(
        "-p",
        "--pos_label",
        metavar="POS",
        default="Relevant",
        help="The label that denotes the positive class",
    )
    parser.add_argument(
        "-n",
        "--neg_label",
        metavar="NEG",
        default="Irrelevant",
        help="The label that denotes the negative class",
    )
    parser.add_argument("-i", "--topic", default=None, type=str)
    args = parser.parse_args()
    if args.module == ApplicationModules.BENCHMARK:
        if args.problem == ActiveLearningTasks.TAR:
            dataset = TarDataset(args.datasettype, args.dataset, args.topic)
            app.tar_benchmark(
                dataset,
                args.target,
                args.exp_choice,
                args.pos_label,
                args.neg_label,
                args.stopinterval,
                args.verboseplotting,
                args.seed
            )
    else:
        raise NotImplementedError("Other tasks have no Benchmark procedure yet")
    
