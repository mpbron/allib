from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import UUID
from instancelib import TextInstance
from instancelib.instances.text import MemoryTextInstance
from instancelib.environment.text import TextEnvironment

from instancelib.ingest.qrel import TrecDataset, hidden
from numpy._typing import NDArray

from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from .reviews import read_review_dataset
import numpy.typing as npt
from ..utils.io import create_dir_if_not_exists
from .datasets import DatasetType, TrecDatasetAL, ReviewDatasets, detect_type
from ..configurations.catalog import ExperimentCombination

import stat

SMALL_MAX = 15000
LARGE_MAX = 50000
PREVALENCE_MIN = 10
EXPERIMENTS_LOW_MEMORY = (
    ExperimentCombination.AUTOTAR,
    ExperimentCombination.CHAO,
    ExperimentCombination.CMH,
    ExperimentCombination.TARGET,
)
EXPERIMENTS_HIGH_MEMORY = (ExperimentCombination.AUTOSTOP,)

EXPERIMENTS_LOW_MEMORY_FILENAME = "experiments_low_memory.txt"
EXPERIMENTS_HIGH_MEMORY_FILENAME = "experiments_high_memory.txt"

SMALL = "topics_small"
LARGE = "topics_large"
XLARGE = "topics_xlarge"


def divide_datasets(
    source_path: Path,
    jobpath: Path,
    dstype: DatasetType,
    pos_label: str,
    small_max=SMALL_MAX,
    large_max=LARGE_MAX,
    prevalence_min=PREVALENCE_MIN,
) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
    dss_name = source_path.stem
    dstype = detect_type(source_path)
    small_path = jobpath / f"{SMALL}_{dss_name}_{dstype}.txt"
    large_path = jobpath / f"{LARGE}_{dss_name}_{dstype}.txt"
    xlarge_path = jobpath / f"{XLARGE}_{dss_name}_{dstype}.txt"
    if dstype == DatasetType.REVIEW:
        dss = ReviewDatasets.from_path(source_path)
    else:
        dss = TrecDatasetAL.from_path(source_path)
    small_topics: List[str] = list()
    large_topics: List[str] = list()
    xlarge_topics: List[str] = list()
    for topic in dss.topic_keys:
        env = dss.get_env(topic)
        size = len(env.dataset)
        size_pos = env.truth.document_count(pos_label)
        if size_pos > prevalence_min:
            if size < small_max:
                small_topics.append(topic)
            elif size < large_max:
                large_topics.append(topic)
            else:
                xlarge_topics.append(topic)
    with small_path.open("w") as fh:
        fh.write("\n".join(small_topics))
    with large_path.open("w") as fh:
        fh.write("\n".join(large_topics))
    with xlarge_path.open("w") as fh:
        fh.write("\n".join(xlarge_topics))
    return small_topics, large_topics, xlarge_topics


def write_experiments_files(
    jobpath: Path,
    low_exps: Sequence[ExperimentCombination] = EXPERIMENTS_LOW_MEMORY,
    high_exps: Sequence[ExperimentCombination] = EXPERIMENTS_HIGH_MEMORY,
):
    low_path = jobpath / EXPERIMENTS_LOW_MEMORY_FILENAME
    high_path = jobpath / EXPERIMENTS_HIGH_MEMORY_FILENAME
    with low_path.open("w") as fh:
        fh.write("\n".join(low_exps))
    with high_path.open("w") as fh:
        fh.write("\n".join(high_exps))


def create_jobs(
    path: Path,
    target_path: Path,
    jobpath: Path,
    pos_label: str,
    small_max=SMALL_MAX,
    large_max=LARGE_MAX,
    prevalence_min=PREVALENCE_MIN,
    test_iterations=30,
    high_cpu: int = 40,
    medium_cpu: int = 20,
    low_cpu: int = 2,
):
    create_dir_if_not_exists(jobpath)
    write_experiments_files(jobpath)
    exp_low_path = (jobpath / EXPERIMENTS_LOW_MEMORY_FILENAME).resolve()
    exp_high_path = (jobpath / EXPERIMENTS_HIGH_MEMORY_FILENAME).resolve()
    dstype = detect_type(path)
    ds_job = jobpath / f"job_{path.stem}_{dstype}.sh"
    dss_name = path.stem
    sj, lj, xlj = divide_datasets(
        path, jobpath, dstype, pos_label, small_max, large_max, prevalence_min
    )
    small_path = (jobpath / f"{SMALL}_{dss_name}_{dstype}.txt").resolve()
    large_path = (jobpath / f"{LARGE}_{dss_name}_{dstype}.txt").resolve()
    xlarge_path = (jobpath / f"{XLARGE}_{dss_name}_{dstype}.txt").resolve()
    if dstype == DatasetType.TREC:
        cmd = "run_trec.sh"
    else:
        cmd = "run_review.sh"
    sp = path.resolve()
    tp = target_path.resolve()
    cmds = [
        "#!/bin/bash",
        f"{cmd} {sp} {small_path} {tp} {exp_low_path} {test_iterations} {high_cpu}",
        f"{cmd} {sp} {small_path} {tp} {exp_high_path} {test_iterations} {medium_cpu}",
        f"{cmd} {sp} {large_path} {tp} {exp_low_path} {test_iterations} {medium_cpu}",
        f"{cmd} {sp} {large_path} {tp} {exp_low_path} {test_iterations} {low_cpu}",
        f"{cmd} {sp} {xlarge_path} {tp} {exp_low_path} {test_iterations} {medium_cpu}",
        f"{cmd} {sp} {xlarge_path} {tp} {exp_high_path} {test_iterations} {low_cpu}",
    ]
    with ds_job.open("w") as fh:
        fh.write("\n".join(cmds))
    ds_job.chmod(ds_job.stat().st_mode | stat.S_IEXEC)
