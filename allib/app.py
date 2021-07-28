from allib.stopcriterion.catalog import StopCriterionCatalog
from allib.configurations.catalog import EstimationConfiguration
from .configurations.base import ESTIMATION_REPOSITORY, STOP_REPOSITORY
from allib.analysis.plotter import BinaryPlotter
from pathlib import Path

from os import PathLike
from .benchmarking.reviews import benchmark
from .configurations import ALConfiguration, FEConfiguration, AL_REPOSITORY, FE_REPOSITORY
from uuid import uuid4

import pandas as pd

def run_benchmark(dataset_path: PathLike, 
                  target_dir: PathLike, 
                  al_choice: str, 
                  fe_choice: str,
                  estimation_choice: str,
                  stop_choice: str
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
    result, plot = benchmark(dataset_path, 
                             al_config, 
                             fe_config, 
                             estimation_config,
                             stop_constructor, 
                             uuid)

    target_path = Path(target_dir)
    
    # Save the benchmark results (or append if the file already exists)
    target_csv = target_path / "benchmark_results.csv"
    if target_csv.exists():
        result_df = pd.read_csv(target_csv, index_col=0)
        result_df = result_df.append([result], ignore_index=True) # type: ignore
    else:
        result_df = pd.DataFrame(data=[result]) # type: ignore
    result_df.to_csv(target_csv)
    
    # Save the plot table
    df_filename = target_path / f"run_{uuid}.csv"
    plot_filename = target_path / f"run_{uuid}.pdf"
    assert isinstance(plot, BinaryPlotter)
    plot.result_frame.to_csv(df_filename)

    plot.show(filename=plot_filename)


    
