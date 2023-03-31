import logging
import pickle
from typing import Optional
from lenses import lens
from pathlib import Path
from uuid import uuid4

from instancelib.utils.func import list_unzip

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
) -> None:
    LOGGER.info(
        f"Start Experiment on {dataset.path.stem} for topic ´{dataset.topic}´ with {exp_choice}"
    )
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
    uuid = uuid4()
    target_path = Path(target_path)
    dataset_name = (
        dataset.path.stem
        if dataset.type == DatasetType.REVIEW or dataset.topic is None
        else f"{dataset.path.stem}-{dataset.topic}"
    )
    # File locations for the plotter object
    dataset_dir = target_path / dataset_name
    plot_filename_pkl = dataset_dir / f"run_{uuid}.pkl"
    plot_filename_pdf = dataset_dir / f"run_{uuid}.pdf"

    # Load the dataset
    create_dir_if_not_exists(dataset_dir)
    plot = benchmark(
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
    )
    with plot_filename_pkl.open("wb") as fh:
        pickle.dump(plot, fh)
    plot.show(filename=plot_filename_pdf)
