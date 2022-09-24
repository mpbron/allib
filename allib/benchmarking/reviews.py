import functools
from dataclasses import dataclass
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID

import numpy as np
import pandas as pd
from instancelib import TextInstance
from instancelib.ingest.spreadsheet import read_csv_dataset

from allib.analysis.experiments import ExperimentIterator
from allib.analysis.tarplotter import ModelStatsTar, TarExperimentPlotter
from allib.configurations.base import STOP_REPOSITORY
from allib.stopcriterion.catalog import StopCriterionCatalog

from ..analysis.analysis import process_performance
from ..analysis.initialization import SeparateInitializer
from ..analysis.plotter import AbstractPlotter, BinaryPlotter
from ..analysis.simulation import TarSimulator, initialize, simulate
from ..stopcriterion.base import AbstractStopCriterion
from ..environment import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.base import AbstractEstimator
from ..estimation.rasch import ParametricRasch
from ..estimation.rasch_python import EMRaschRidgePython
from ..module.factory import MainFactory
from ..utils.func import list_unzip3

POS = "Relevant"
NEG = "Irrelevant"


def binary_mapper(value: Any) -> str:
    return POS if value == 1 else NEG


DLT = TypeVar("DLT")
LT = TypeVar("LT")


def read_review_dataset(
    path: "PathLike[str]",
) -> AbstractEnvironment[
    TextInstance[Union[int, UUID], np.ndarray],
    Union[int, UUID],
    str,
    np.ndarray,
    str,
    str,
]:
    """Convert a CSV file with a Systematic Review dataset to a MemoryEnvironment.

    Parameters
    ----------
    path : PathLike
        The path to the CSV file

    Returns
    -------
    MemoryEnvironment[int, str, np.ndarray, str]
        A MemoryEnvironment. The labels that
    """
    df = pd.read_csv(path)
    if "label_included" in df.columns:
        env = read_csv_dataset(
            path,
            data_cols=["title", "abstract"],
            label_cols=["label_included"],
            label_mapper=binary_mapper,
        )
    else:
        env = read_csv_dataset(
            path,
            data_cols=["title", "abstract"],
            label_cols=["included"],
            label_mapper=binary_mapper,
        )
    al_env = MemoryEnvironment.from_instancelib_simulation(env)
    return al_env


@dataclass
class BenchmarkResult:
    dataset: PathLike
    uuid: UUID
    stop_wss: Mapping[str, float]
    stop_recall: Mapping[str, float]
    stop_loss_er: Mapping[str, float]
    stop_effort: Mapping[str, int]
    stop_prop_effort: Mapping[str, float]


def benchmark(
    path: PathLike,
    uuid: UUID,
    al_config: Dict[str, Any],
    fe_config: Dict[str, Any],
    estimators: Mapping[str, AbstractEstimator[Any, Any, Any, Any, Any, str]],
    stopcriteria: Mapping[str, AbstractStopCriterion[str]],
) -> Tuple[BenchmarkResult, TarExperimentPlotter[str]]:
    env = read_review_dataset(path)
    factory = MainFactory()
    initializer = SeparateInitializer(env, 1)
    al, _ = initialize(factory, al_config, fe_config, initializer, env)
    exp = ExperimentIterator(al, POS, NEG, stopcriteria, estimators, 10, 10, 10)
    plotter = ModelStatsTar(POS, NEG)
    simulator = TarSimulator(exp, plotter)
    simulator.simulate()
    # Criterion results
    stop_wss = {crit: plotter.wss_at_stop(crit) for crit in stopcriteria}
    stop_recall = {crit: plotter.recall_at_stop(crit) for crit in stopcriteria}
    stop_loss_er = {crit: plotter.loss_er_at_stop(crit) for crit in stopcriteria}
    stop_effort = {crit: plotter.effort_at_stop(crit) for crit in stopcriteria}
    stop_prop_effort = {
        crit: plotter.proportional_effort_at_stop(crit) for crit in stopcriteria
    }
    result = BenchmarkResult(
        path, uuid, stop_wss, stop_recall, stop_loss_er, stop_effort, stop_prop_effort
    )
    return result, plotter
