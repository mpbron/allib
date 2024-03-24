import itertools
import operator
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Optional, Tuple, TypeVar
from uuid import UUID

import pandas as pd

from allib.analysis.analysis import loss_er
from allib.configurations.base import ExperimentCombination

from tqdm.auto import tqdm

from .statistics import TarDatasetStats
from .tarplotter import TarExperimentPlotter

_T = TypeVar("_T")


@dataclass
class BenchmarkResult:
    dataset: str
    run_id: UUID
    seed: Optional[int]
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
                splitted = file.stem.split("_")
                run_id = UUID(splitted[1])
                # seed = int(splitted[2])
                yield (run_id, plotter)

    dss = {ds_path.stem: dict(read_ds(ds_path)) for ds_path in root_path.iterdir()}
    return dss


def nest(tupiter: Iterable[Tuple[_T, ...]]) -> Mapping[_T, Any]:
    def value(k, iterable: Iterable[Tuple[_T, ...]]) -> Tuple[_T, Any]:
        tuplist = tuple(iterable)
        if len(tuplist[0]) > 2:
            return k, nest(tup[1:] for tup in tuplist)
        return tuplist[-1]

    structured = dict(
        (
            value(k, tups)
            for (k, tups) in itertools.groupby(tupiter, operator.itemgetter(0))
        )
    )
    return structured


def read_results(
    root_path: Path,
) -> Mapping[str, Mapping[str, Mapping[UUID, Mapping[int, TarExperimentPlotter]]]]:
    def read_ds() -> Iterator[Tuple[str, str, UUID, int, TarExperimentPlotter]]:
        for ds_path in tqdm(list(root_path.iterdir())):
            for method in ds_path.iterdir():
                for file in method.iterdir():
                    if file.suffix == ".pkl":
                        with file.open("rb") as fh:
                            plotter: TarExperimentPlotter = pickle.load(fh)
                        splitted = file.stem.split("_")
                        run_id = UUID(splitted[1])
                        seed = int(splitted[2])
                        yield (ds_path.stem, method.stem, run_id, seed, plotter)

    tuples = list(read_ds())
    dss = nest(tuples)
    return dss  # type: ignore

def read_results_memsafe(
    root_path: Path,
) -> pd.DataFrame:
    def read_ds() -> Iterator[BenchmarkResult]:
        for ds_path in tqdm(list(root_path.iterdir())):
            for method in ds_path.iterdir():
                for file in method.iterdir():
                    if file.suffix == ".pkl":
                        with file.open("rb") as fh:
                            plotter: TarExperimentPlotter = pickle.load(fh)
                        splitted = file.stem.split("_")
                        run_id = UUID(splitted[1])
                        seed = int(splitted[2])
                        for crit_name in plotter.criterion_names:
                            yield extract_results(ds_path.stem, run_id, plotter,crit_name, seed=seed)
    records = list(read_ds())
    df = pd.DataFrame(records)
    df = pd.concat((df, df.dataset_stats.apply(pd.Series)), axis=1)
    df = df.drop("dataset_stats", axis=1)
    return df


def extract_results(
    dataset: str,
    run_id: UUID,
    plotter: TarExperimentPlotter,
    crit_name: str,
    target_recall: float = 0.95,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    stop_it = plotter._it_at_stop(crit_name) or plotter.it
    rs = plotter.recall_stats[stop_it]
    ds = plotter.dataset_stats[stop_it]
    re = abs(target_recall - rs.recall) / target_recall
    return BenchmarkResult(
        dataset,
        run_id,
        seed,
        rs.name,
        crit_name,
        rs.wss,
        rs.recall,
        rs.loss_er,
        rs.pos_docs_found,
        rs.effort,
        rs.proportional_effort,
        re,
        ds,
    )


def extract_information_old_format(
    run_dict: Mapping[str, Mapping[UUID, TarExperimentPlotter]]
) -> Sequence[BenchmarkResult]:
    records = [
        extract_results(dataset, run_id, plotter, crit_name)
        for (dataset, runs) in run_dict.items()
        for (run_id, plotter) in runs.items()
        for crit_name in plotter.criterion_names
    ]
    return records


def extract_information(
    run_dict: Mapping[
        str, Mapping[str, Mapping[UUID, Mapping[int, TarExperimentPlotter]]]
    ]
) -> Sequence[BenchmarkResult]:
    records = [
        extract_results(dataset, run_id, plotter, crit_name, seed=seed)
        for (dataset, learner_method) in run_dict.items()
        for (learner, runs) in learner_method.items()
        for (run_id, run_seeds) in runs.items()
        for (seed, plotter) in run_seeds.items()
        for crit_name in plotter.criterion_names
    ]
    return records


def results_to_pandas(results: Sequence[BenchmarkResult]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = pd.concat((df, df.dataset_stats.apply(pd.Series)), axis=1)
    df = df.drop("dataset_stats", axis=1)
    return df


def read_plot(
    source: Path,
    dataset: str,
    method: ExperimentCombination,
    seed: int,
    uuid: Optional[UUID] = None,
) -> TarExperimentPlotter:
    if uuid is not None:
        filepath = source / dataset / str(method) / f"run_{uuid}_{seed}.pkl"
    else:
        exp_dir = source / dataset / str(method)
        files = (
            file
            for file in exp_dir.iterdir()
            if int(file.stem.split("_")[-1]) == seed and file.suffix == ".pkl"
        )
        # pick first
        filepath = next(files)
    with filepath.open("rb") as fh:
        obj: TarExperimentPlotter = pickle.load(fh)
    return obj
