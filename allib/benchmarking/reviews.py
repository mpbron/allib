import functools
from dataclasses import dataclass
from os import PathLike
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Sequence,
                    Set, Tuple, TypeVar, Union)
from uuid import UUID

import numpy as np
import pandas as pd
from instancelib import TextInstance
from instancelib.ingest.spreadsheet import read_csv_dataset

from ..analysis.analysis import process_performance
from ..analysis.initialization import SeparateInitializer
from ..analysis.plotter import AbstractPlotter, BinaryPlotter
from ..analysis.simulation import initialize, simulate
from ..analysis.stopping import RaschCaptureCriterion
from ..configurations.base import AL_REPOSITORY, FE_REPOSITORY
from ..configurations.catalog import ALConfiguration, FEConfiguration
from ..environment import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.rasch import ParametricRasch
from ..module.factory import MainFactory
from ..utils.func import list_unzip3


def binary_mapper(value: Any) -> str:
    if value == 1:
        return "Relevant"
    return "Irrelevant"
    

DLT = TypeVar("DLT")
LT = TypeVar("LT")

def read_review_dataset(path: "PathLike[str]") -> AbstractEnvironment[
    TextInstance[int, np.ndarray], Union[int, UUID], str, np.ndarray, str, str]:
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
        env = read_csv_dataset(path, 
                               data_cols=["title", "abstract"], 
                               label_cols=["label_included"], 
                               label_mapper=binary_mapper)
    else:
         env = read_csv_dataset(path, data_cols=["title", "abstract"], 
                               label_cols=["included"], 
                               label_mapper=binary_mapper)
    al_env = MemoryEnvironment.from_instancelib_simulation(env)
    return al_env

@dataclass
class BenchmarkResult:
    dataset: PathLike
    uuid: UUID
    wss: float
    recall: float

def benchmark(path: PathLike, 
              al_config: Dict[str, Any], 
              fe_config: Dict[str, Any], 
              uuid: UUID) -> Tuple[BenchmarkResult, AbstractPlotter[str]]:
    """Run a single benchmark test for the given configuration

    TODO: Parametrize Stopping criteria 
    TODO: Parametrize initialization
    TODO: Parametrize labels

    Parameters
    ----------
    path : PathLike
        The path of the dataset
    al_setup : ALConfiguration
        One of the ALConfiguration Enum members
    fe_setup : FEConfiguration
        One of the FEConfiguration Enum members

    Returns
    -------
    Tuple[BenchmarkResult, BinaryPlotter[str]]
        A tuple containing:

        - The result of the Benchmark
        - The plot of the run

    """    
    environment = read_review_dataset(path)
    
    # Create the components
    factory = MainFactory()
    # TODO: Enable creation from parameters
    initializer = SeparateInitializer(environment, 1)
    rasch = ParametricRasch[int, str, np.ndarray, str, str]()
    stop = RaschCaptureCriterion[str](rasch, "Relevant", 3, 1.0)
    
    # Simulate the annotation workflow
    plotter = BinaryPlotter[str]("Relevant", "Irrelevant", rasch)
    al, _ = initialize(factory, al_config, fe_config, initializer, environment)
    al_result, plotter_result = simulate(al, stop, plotter, 10)

    # Assess the performance
    performance = process_performance(al_result, "Relevant")
    result = BenchmarkResult(path, uuid, 
                             performance.wss, performance.recall)
    return result, plotter_result
