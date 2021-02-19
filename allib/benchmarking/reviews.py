from uuid import UUID
from allib.analysis.analysis import process_performance
from allib.analysis.plotter import BinaryPlotter
from allib.estimation.rasch import ParametricRasch
from allib.analysis.stopping import RaschCaptureCriterion
from allib.analysis.initialization import SeparateInitializer
from allib.module.factory import MainFactory
from allib.configurations.catalog import ALConfiguration, FEConfiguration
from allib.configurations.base import AL_REPOSITORY, FE_REPOSITORY
import functools
from os import PathLike
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Sequence, Set, Tuple, TypeVar

import numpy as np
import pandas as pd

from dataclasses import dataclass

from ..analysis.simulation import initialize, simulate
from ..environment import MemoryEnvironment
from ..utils.func import list_unzip3

BINARY_MAPPING = {
    0: "Irrelevant",
    1: "Relevant"
}

DLT = TypeVar("DLT")
LT = TypeVar("LT")

def inv_transform_mapping(column_name: str, 
                          row: pd.Series, 
                          label_mapping: Mapping[int, str] = BINARY_MAPPING
                          ) -> FrozenSet[str]:
    """Convert the numeric coded label in column `column_name` in row `row`
    to a string according to the mapping in `label_mapping`.

    Parameters
    ----------
    column_name : str
        The column in which the labels are stored
    row : pd.Series
        A row from a Pandas DataFrame
    label_mapping : Mapping[int, str], optional
        A mapping from integer to strings, by default BINARY_MAPPING
        that maps 0 => Irrelevant and 1 => Relevant

    Returns
    -------
    FrozenSet[str]
        A set of labels that belong to the row
    """    
    coded_label: int = row[column_name]
    if coded_label in label_mapping:
        readable_label = label_mapping[coded_label] 
        return frozenset([readable_label])
    return frozenset([])

def extract_data(dataset_df: pd.DataFrame, 
               data_cols: Sequence[str], 
               labelfunc: Callable[..., FrozenSet[str]]
               ) -> Tuple[List[int], List[str], List[FrozenSet[str]]]:
    """Extract text data and labels from a dataframe

    Parameters
    ----------
    dataset_df : pd.DataFrame
        The dataset
    data_cols : List[str]
        The cols in which the text is stored
    labelfunc : Callable[..., FrozenSet[str]]
        A function that maps rows to sets of labels

    Returns
    -------
    Tuple[List[int], List[str], List[FrozenSet[str]]]
        [description]
    """    
    def yield_row_values():
        for i, row in dataset_df.iterrows():
            data = " ".join([str(row[col]) for col in data_cols])
            labels = labelfunc(row)
            yield int(i), str(data), labels  # type: ignore
    indices, texts, labels_true = list_unzip3(yield_row_values())
    return indices, texts, labels_true  # type: ignore

def build_environment(df: pd.DataFrame, 
                      label_mapping: Mapping[Any, str],
                      data_cols: Sequence[str],
                      label_col: str,
                     ) -> MemoryEnvironment[int, str, np.ndarray, str]:
    """Build an environment from a data frame

    Parameters
    ----------
    df : pd.DataFrame
        A data frame that contains all texts and labels
    label_mapping : Mapping[int, str]
        A mapping from indices to label strings
    data_cols : Sequence[str]
        A sequence of columns that contain the texts
    label_col : str
        The name of the column that contains the label data

    Returns
    -------
    MemoryEnvironment[int, str, np.ndarray, str]
        A MemoryEnvironment that contains the  
    """    
    labelfunc = functools.partial(inv_transform_mapping, label_col, label_mapping=label_mapping)
    indices, texts, true_labels = extract_data(df, data_cols, labelfunc)
    environment = MemoryEnvironment[int, str, np.ndarray, str].from_data(
        label_mapping.values(), 
        indices, texts, true_labels,
        [])
    return environment

def read_review_dataset(path: PathLike) -> MemoryEnvironment[int, str, np.ndarray, str]:
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
        env = build_environment(df, BINARY_MAPPING, data_cols=["title", "abstract"], label_col="label_included")
        return env
    env = build_environment(df, BINARY_MAPPING, data_cols=["title", "abstract"], label_col="included")
    return env

@dataclass
class BenchmarkResult:
    dataset: PathLike
    uuid: UUID
    wss: float
    recall: float

def benchmark(path: PathLike, 
              al_config: Dict[str, Any], 
              fe_config: Dict[str, Any], 
              uuid: UUID) -> Tuple[BenchmarkResult, BinaryPlotter[str]]:
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
    plotter = BinaryPlotter[str]("Relevant", "Irrelevant")
    al, _ = initialize(factory, al_config, fe_config, initializer, environment)
    al_result, plotter_result = simulate(al, stop, plotter, 10)

    # Assess the performance
    performance = process_performance(al_result, "Relevant")
    result = BenchmarkResult(path, uuid, 
                             performance.wss, performance.accuracy)
    return result, plotter_result