import pickle
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.stopcriterion.catalog import StopCriterionCatalog
from allib.configurations.catalog import EstimationConfiguration
from .configurations.base import ESTIMATION_REPOSITORY, STOP_REPOSITORY
from allib.analysis.plotter import BinaryPlotter
from pathlib import Path

from .benchmarking.reviews import benchmark
from .configurations import (
    ALConfiguration,
    FEConfiguration,
    AL_REPOSITORY,
    FE_REPOSITORY,
)
from uuid import uuid4

import pandas as pd


def run_benchmark(
    dataset_path: Path,
    target_path: Path,
    al_choice: str,
    fe_choice: str,
    estimation_choice: str,
    stop_choice: str,
) -> None:
    # Parse Configuration
    al_setup = ALConfiguration(al_choice)
    fe_setup = FEConfiguration(fe_choice)
    estimation_setup = EstimationConfiguration(estimation_choice)
    stop_setup = StopCriterionCatalog(stop_choice)

    # Retrieve Configuration
    al_config = AL_REPOSITORY[al_setup]
    fe_config = FE_REPOSITORY[fe_setup]
    estimation_config = ESTIMATION_REPOSITORY[estimation_setup]
    stop_constructor = STOP_REPOSITORY[stop_setup]

    # Run Benchmark
    uuid = uuid4()
    result, plot = benchmark(
        dataset_path, uuid, al_config, fe_config, estimators={}, stopcriteria={}
    )

    target_path = Path(target_path)

    # Save the benchmark results (or append if the file already exists)
    target_file = target_path / "benchmark_results.pkl"
    if target_file.exists():
        result_df = pd.read_pickle(target_file)
        new_result = pd.DataFrame([result])
        result_df = pd.concat([result_df, new_result], ignore_index=True)  # type: ignore
    else:
        result_df = pd.DataFrame(data=[result])  # type: ignore
    result_df.to_pickle(target_file)

    # Save the plot table
    df_filename_csv = target_path / f"run_{uuid}.pkl"
    df_filename_pkl = target_path / f"run_{uuid}.csv"

    plot_filename = target_path / f"run_{uuid}.pdf"
    assert isinstance(plot, TarExperimentPlotter)
    with df_filename_pkl.open("wb") as fh:
        pickle.dump(plot, fh)
    plot.show(filename=plot_filename)
