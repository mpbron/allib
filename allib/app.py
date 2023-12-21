import logging
import pickle
from typing import Any, Optional, Tuple
from lenses import lens
from pathlib import Path
from uuid import uuid4

from instancelib.utils.func import list_unzip

from .activelearning.base import ActiveLearner
from .analysis.summarize import read_plot

from .analysis.tarplotter import TarExperimentPlotter
from .benchmarking.datasets import TarDataset, DatasetType
from .benchmarking.reviews import benchmark, read_review_dataset
from .configurations import AL_REPOSITORY, FE_REPOSITORY
from .configurations.base import (
    EXPERIMENT_REPOSITORY,
    STOP_BUILDER_REPOSITORY,
    TarExperimentParameters,
)
from .configurations.catalog import ExperimentCombination
from .utils.func import flatten_dicts
from .utils.io import create_dir_if_not_exists

LOGGER = logging.getLogger(__name__)



def tar_benchmark(
    dataset: TarDataset,
    target_path: Path,
    exp_choice: ExperimentCombination,
    pos_label: str,
    neg_label: str,
    stop_interval: Optional[int] = None,
    enable_plots=True,
    seed: Optional[int] = None,
    max_it: Optional[int] = None
) -> Tuple[Optional[ActiveLearner[Any, Any, Any, Any, Any, str]],TarExperimentPlotter[str]]:
    exp = EXPERIMENT_REPOSITORY[exp_choice]
    # Overrride stop and estimation intervals if desired
    if stop_interval is not None:
        l1 = lens.estimation_interval.set(stop_interval)
        l2 = lens.stop_interval.set(stop_interval)
        exp: TarExperimentParameters = l1(l2(exp))

    # Retrieve Configuration
    al_config = AL_REPOSITORY[exp.al_configuration]
    fe_config = (
        dict() if exp.fe_configuration is None else FE_REPOSITORY[exp.fe_configuration]
    )
    stop_builders = [
        STOP_BUILDER_REPOSITORY[config] for config in exp.stop_builder_configuration
    ]
    initializer = exp.init_configuration
    estimator_dicts, stop_criteria_dicts = list_unzip(
        map(lambda f: f(pos_label, neg_label), stop_builders)
    )
    estimators = flatten_dicts(*estimator_dicts)
    stop_criteria = flatten_dicts(*stop_criteria_dicts)

    # Specify benchmark targets and outputs
    run_id = uuid4()
    target_path = Path(target_path)
    dataset_name = (
        dataset.path.stem
        if dataset.type == DatasetType.REVIEW or dataset.topic is None
        else f"{dataset.path.stem}-{dataset.topic}"
    )
    # File locations for the plotter object
    dataset_dir = target_path / dataset_name
    method_dir = dataset_dir / str(exp_choice)
    finished_file = method_dir / f"run_{seed}.finished"
    if finished_file.exists() and seed is not None:
        return None, read_plot(target_path, dataset_name, exp_choice, seed)
    create_dir_if_not_exists(method_dir)
    plot_filename_pkl = method_dir / f"run_{run_id}_{seed}.pkl"
    plot_filename_pdf = method_dir / f"run_{run_id}_{seed}.pdf"
    # Load the dataset
    create_dir_if_not_exists(dataset_dir)
    print(f"Starting simulation for method {exp_choice} on dataset {dataset_name} with seed {seed}. Run id = {run_id}")
    al, plot = benchmark(
        dataset.env,
        plot_filename_pkl,
        plot_filename_pdf,
        al_config,
        fe_config,
        initializer,
        estimators,
        stop_criteria,
        pos_label,
        neg_label,
        batch_size=exp.batch_size,
        stop_interval=exp.stop_interval,
        estimation_interval=exp.estimation_interval,
        enable_plots=enable_plots,
        seed=seed,
        max_it=max_it
    )
    with plot_filename_pkl.open("wb") as fh:
        pickle.dump(plot, fh)
    if plot.it > 1:
        plot.show(filename=plot_filename_pdf)
    if seed is not None and len(al.env.labeled) == len(al.env.dataset):
        finished_file.touch()
    return al, plot
