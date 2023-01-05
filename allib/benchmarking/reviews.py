import logging
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar, Union
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pandas as pd
import instancelib as il
from instancelib import TextInstance
from instancelib.ingest.spreadsheet import read_csv_dataset
from instancelib.typehints.typevars import KT, LT
from ..typehints.typevars import IT

from ..analysis.experiments import ExperimentIterator
from ..analysis.initialization import Initializer, SeparateInitializer
from ..analysis.simulation import TarSimulator, initialize_tar_simulation
from ..analysis.tarplotter import ModelStatsTar, TarExperimentPlotter
from ..environment import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.base import AbstractEstimator
from ..module.factory import MainFactory
from ..stopcriterion.base import AbstractStopCriterion
from ..utils.func import list_unzip3

POS = "Relevant"
NEG = "Irrelevant"
LOGGER = logging.getLogger(__name__)


def binary_mapper(value: Any) -> str:
    return POS if value == 1 else NEG


DLT = TypeVar("DLT")
LT = TypeVar("LT")


def read_review_dataset(
    path: Path,
) -> AbstractEnvironment[
    TextInstance[int, npt.NDArray[Any]],
    Union[int, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
    """Convert a CSV file with a Systematic Review dataset to a MemoryEnvironment.

    Parameters
    ----------
    path : Path
        The path to the CSV file

    Returns
    -------
    MemoryEnvironment[int, str, npt.NDArray[Any], str]
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
    if isinstance(env, il.TextEnvironment):
        env = env.shuffle(env)
    al_env = MemoryEnvironment.from_instancelib_simulation(env)
    return al_env  # type: ignore


def benchmark(
    env: AbstractEnvironment[IT, KT, Any, Any, Any, str],
    output_path: Path,
    output_pdf_path: Path,
    al_config: Mapping[str, Any],
    fe_config: Mapping[str, Any],
    initializer_builder: Callable[..., Initializer[IT, KT, str]],
    estimators: Mapping[str, AbstractEstimator[IT, KT, Any, Any, Any, str]],
    stopcriteria: Mapping[str, AbstractStopCriterion[str]],
    pos_label: str,
    neg_label: str,
    batch_size: int = 10,
    stop_interval: Union[int, Mapping[str, int]] = 10,
    estimation_interval: Union[int, Mapping[str, int]] = 10,
) -> TarExperimentPlotter[str]:
    factory = MainFactory()
    initializer = initializer_builder(pos_label=pos_label, neg_label=neg_label)
    al, _ = initialize_tar_simulation(
        factory, al_config, fe_config, initializer, env, pos_label, neg_label
    )
    exp = ExperimentIterator(
        al,
        pos_label,
        neg_label,
        stopcriteria,
        estimators,
        batch_size,
        stop_interval,
        estimation_interval,
    )
    plotter = ModelStatsTar(POS, NEG)
    simulator = TarSimulator(
        exp, plotter, output_path=output_path, output_pdf_path=output_pdf_path
    )
    try:
        simulator.simulate()
    except Exception as e:
        LOGGER.error("Exited with %s", e)
        pass
    return plotter
