import numpy as np
from typing import Sequence, Dict, List, Any, Callable, TypeVar
from sklearn.base import clone
from .base import AbstractActiveLearner
from sklearn.base import BaseEstimator

from .datapoint import DataPoint
from .oracles import OracleFunction

T = TypeVar('T')

def benchmark(
        al_model: AbstractActiveLearner,
        f_oracle: OracleFunction,
        f_metric: Callable[[np.ndarray, np.ndarray], T],
        x_test: np.ndarray, 
        y_test: np.ndarray,
        n_iterations: int,
        n_initialize: int) -> List[T]:
    al_model.initialize(f_oracle, n_initialize)
    al_model.fit()
    accs = []
    for _ in range(n_iterations):
        if al_model.len_unlabeled() > 0:
            al_model.iterate_al(f_oracle)
            y_test_pred = al_model.estimator.predict(x_test)
            accs.append(f_metric(y_test, y_test_pred))
    return accs


def al_benchmarking(
        al_models: Dict[str, AbstractActiveLearner], 
        labels: List[str],
        datapoints: List[DataPoint],
        f_oracle: OracleFunction, 
        f_metric: Callable[[np.ndarray, np.ndarray], T],
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_iterations: int, 
        n_initialize: int) -> Dict[str, List[T]]:
    metrics_dict = {al_id: benchmark(
            al_model,
            f_oracle, f_metric,
            x_test, y_test,
            n_iterations, n_initialize) for al_id, al_model in al_models.items()}
    return metrics_dict
