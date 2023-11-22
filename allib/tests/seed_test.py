from .base_test import POS, NEG, DATASET
from pathlib import Path

from ..app import tar_benchmark
from ..benchmarking.datasets import DatasetType, TarDataset
from ..configurations.catalog import ExperimentCombination

def test_seed(tmp_path: Path):
    dataset = TarDataset(DatasetType.REVIEW, DATASET, None)
    target_path = tmp_path / "results"
    at, _ = tar_benchmark(dataset, target_path, ExperimentCombination.AUTOTAR, POS, NEG, seed=42, max_it=-1)
    assert at is not None
    ens, _ = tar_benchmark(dataset, target_path, ExperimentCombination.CHAO_IB, POS, NEG, seed=42, max_it=-1)
    assert ens is not None
    assert frozenset(at.env.labeled) == frozenset(ens.env.labeled) 
    
def test_no_seed(tmp_path: Path):
    dataset = TarDataset(DatasetType.REVIEW, DATASET, None)
    target_path = tmp_path / "results"
    at, _ = tar_benchmark(dataset, target_path, ExperimentCombination.AUTOTAR, POS, NEG, seed=None, max_it=-1)
    assert at is not None
    ens, _ = tar_benchmark(dataset, target_path, ExperimentCombination.CHAO_IB, POS, NEG, seed=None, max_it=-1)
    assert ens is not None
    assert frozenset(at.env.labeled) != frozenset(ens.env.labeled) 