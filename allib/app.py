import logging
import pickle
from pathlib import Path
from uuid import uuid4

from instancelib.utils.func import list_unzip

from .analysis.tarplotter import TarExperimentPlotter
from .benchmarking.reviews import benchmark, read_review_dataset
from .configurations import AL_REPOSITORY, FE_REPOSITORY
from .configurations.base import EXPERIMENT_REPOSITORY, STOP_BUILDER_REPOSITORY
from .configurations.catalog import ExperimentCombination
from .utils.func import flatten_dicts
from .utils.io import create_dir_if_not_exists

LOGGER = logging.getLogger(__name__)


def tar_benchmark(
    dataset_path: Path,
    target_path: Path,
    exp_choice: ExperimentCombination,
    pos_label: str,
    neg_label: str,
) -> None:
    LOGGER.info(f"Start Experiment on {dataset_path} with {exp_choice}")
    exp = EXPERIMENT_REPOSITORY[exp_choice]

    # Parse Configuration

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
    dataset_name = dataset_path.stem
    # File locations for the plotter object
    dataset_dir = target_path / dataset_name
    plot_filename_pkl = dataset_dir / f"run_{uuid}.pkl"
    plot_filename_pdf = dataset_dir / f"run_{uuid}.pdf"

    # Load the dataset
    env = read_review_dataset(dataset_path)
    create_dir_if_not_exists(dataset_dir)
    plot = benchmark(
        env,
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
    )
    with plot_filename_pkl.open("wb") as fh:
        pickle.dump(plot, fh)
    plot.show(filename=plot_filename_pdf)
