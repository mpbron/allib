from pathlib import Path
import pickle

from ..analysis.experiments import ExperimentIterator
from ..analysis.simulation import TarSimulator, initialize_tar_simulation
from ..analysis.tarplotter import ModelStatsTar
from ..app import tar_benchmark
from ..benchmarking.datasets import DatasetType, TarDataset
from ..configurations.catalog import ExperimentCombination
from ..module.factory import MainFactory
from ..utils.func import flatten_dicts, list_unzip
from ..configurations import AL_REPOSITORY, FE_REPOSITORY
from ..configurations.base import EXPERIMENT_REPOSITORY, STOP_BUILDER_REPOSITORY

POS = "Relevant"
NEG = "Irrelevant"
DATASET = Path(__file__).parent / "testdataset.csv"


def test_autotar(tmp_path: Path):
    dataset = TarDataset(DatasetType.REVIEW, DATASET, None)
    target_path = tmp_path / "results"
    tar_benchmark(dataset, target_path, ExperimentCombination.AUTOTAR, POS, NEG)


def test_autostop(tmp_path: Path):
    dataset = TarDataset(DatasetType.REVIEW, DATASET, None)
    target_path = tmp_path / "results"
    tar_benchmark(dataset, target_path, ExperimentCombination.AUTOSTOP, POS, NEG)


def test_early_terminate(tmp_path: Path):
    factory = MainFactory()
    dataset = TarDataset(DatasetType.REVIEW, DATASET, None)
    config = EXPERIMENT_REPOSITORY[ExperimentCombination.AUTOTAR]

    al_config = AL_REPOSITORY[config.al_configuration]
    fe_config = (
        dict()
        if config.fe_configuration is None
        else FE_REPOSITORY[config.fe_configuration]
    )
    stop_builders = [
        STOP_BUILDER_REPOSITORY[config] for config in config.stop_builder_configuration
    ]
    initializer = config.init_configuration(pos_label=POS, neg_label=POS)
    estimator_dicts, stop_criteria_dicts = list_unzip(
        map(lambda f: f(POS, NEG), stop_builders)
    )
    estimators = flatten_dicts(*estimator_dicts)
    stopcriteria = flatten_dicts(*stop_criteria_dicts)
    al, _ = initialize_tar_simulation(
        factory,
        al_config,
        fe_config,
        initializer,
        dataset.env,
        POS,
        NEG,
    )
    exp = ExperimentIterator(
        al,
        POS,
        NEG,
        stopcriteria,
        estimators,
        1,
        10,
        10,
    )
    plotter = ModelStatsTar(POS, NEG)
    output_path = tmp_path / "plotter.pkl"
    output_pdf_path = tmp_path / "plot.pdf"
    simulator = TarSimulator(
        exp,
        plotter,
        output_path=output_path,
        output_pdf_path=output_pdf_path,
        plot_enabled=True,
        stop_when_found_all=True,
    )
    simulator.simulate()
    with output_path.open("wb") as fh:
        pickle.dump(plotter, fh)
    plotter.show(filename=output_pdf_path)
    learner = simulator.experiment.learner
    assert len(learner.env.labeled) < len(learner.env.dataset)
    assert not learner.env.truth.get_instances_by_label(POS).difference(
        learner.env.labeled
    )
