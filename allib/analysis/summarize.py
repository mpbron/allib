import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import UUID

import pandas as pd

from allib.analysis.analysis import loss_er

from ..utils.func import hn

from .statistics import TarDatasetStats
from .tarplotter import TarExperimentPlotter


@dataclass
class BenchmarkResult:
    dataset: str
    run_id: UUID
    learner_id: str
    stop_criterion: str

    stop_wss: float
    stop_recall: float
    stop_loss_er: float
    stop_found: int
    stop_effort: int
    stop_prop_effort: float
    stop_relative_error: float

    dataset_stats: TarDatasetStats


def read_datasets(root_path: Path) -> Mapping[str, Mapping[UUID, TarExperimentPlotter]]:
    def read_ds(ds_path: Path):
        for file in ds_path.iterdir():
            if file.suffix == ".pkl":
                with file.open("rb") as fh:
                    plotter = pickle.load(fh)
                run_id = UUID(file.stem.split("_")[1])
                yield (run_id, plotter)

    dss = {ds_path.stem: dict(read_ds(ds_path)) for ds_path in root_path.iterdir()}
    return dss


def extract_results(
    dataset: str,
    run_id: UUID,
    plotter: TarExperimentPlotter,
    crit_name: str,
    target_recall: float = 0.95,
) -> BenchmarkResult:
    stop_it = plotter._it_at_stop(crit_name) or plotter.it
    rs = plotter.recall_stats[stop_it]
    ds = plotter.dataset_stats[stop_it]
    l_er = loss_er(rs.pos_docs_found, rs.effort, ds.pos_count, ds.size)
    re = abs(target_recall - rs.recall) / target_recall
    return BenchmarkResult(
        dataset,
        run_id,
        rs.name,
        crit_name,
        rs.wss,
        rs.recall,
        l_er,
        rs.pos_docs_found,
        rs.effort,
        rs.proportional_effort,
        re,
        ds,
    )


def extract_information(
    run_dict: Mapping[str, Mapping[UUID, TarExperimentPlotter]]
) -> Sequence[BenchmarkResult]:
    records = [
        extract_results(dataset, run_id, plotter, crit_name)
        for (dataset, runs) in run_dict.items()
        for (run_id, plotter) in runs.items()
        for crit_name in plotter.criterion_names
    ]
    return records


def results_to_pandas(results: Sequence[BenchmarkResult]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = pd.concat((df, df.dataset_stats.apply(pd.Series)), axis=1)
    df = df.drop("dataset_stats", axis=1)
    return df
