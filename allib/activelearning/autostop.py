import collections
from math import ceil
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Optional,
    Sequence,
    Tuple,
)

import instancelib as il
import numpy as np
import numpy.typing as npt
from instancelib.typehints import DT, KT, LT, RT, VT

from ..environment.base import AbstractEnvironment
from ..typehints import IT
from ..utils.func import list_unzip
from .autotar import AutoTarLearner


def calc_ap_prior_distribution(
    ranking: Sequence[Tuple[KT, float]]
) -> Sequence[Tuple[KT, float]]:
    keys, _ = list_unzip(ranking)
    N = len(ranking)
    ranks = np.array(range(1, (N + 1)))
    Z = np.sum(np.log(N / ranks))
    pis: npt.NDArray[Any] = 1.0 / Z * np.log(N / ranks)
    return list(zip(keys, pis.tolist()))


class AutoStopLearner(
    AutoTarLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    distributions: Dict[int, Dict[KT, float]]

    dist_fixed: Dict[int, npt.NDArray[Any]]
    cumulative_fo: Dict[int, npt.NDArray[Any]]
    cumulative_so: Dict[int, npt.NDArray[Any]]
    label_vector: Dict[int, npt.NDArray[Any]]
    cumulative_sampled: Dict[int, FrozenSet[KT]]
    key_seq: Sequence[KT]

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        pos_label: LT,
        neg_label: LT,
        k_sample: int,
        batch_size: int,
        *_,
        seed: int = 0,
        prune: bool = True,
        identifier: Optional[str] = None,
        **__
    ) -> None:
        super().__init__(
            env,
            classifier,
            pos_label,
            neg_label,
            k_sample,
            batch_size,
            *_,
            seed=seed,
            identifier=identifier,
            **__
        )

        super().__init__(
            env,
            classifier,
            pos_label,
            neg_label,
            k_sample,
            batch_size,
            identifier=identifier,
            seed=seed,
        )
        self.distributions = dict()
        self.dist_fixed = dict()
        self.cumulative_fo = dict()
        self.cumulative_so = dict()
        self.cumulative_sampled = dict()
        self.key_seq = tuple(self.env.dataset)
        self.label_vector = dict()
        self.dtype = np.float128
        self.prune = prune

    def inclusion_probability(self, key: KT, t_max: int) -> float:
        dist_h = self.distributions
        pi = 1.0 - np.product(
            [
                ((1.0 - dist_h[t][key]) ** len(self.sampled_sets[t]))
                for t in dist_h
                if t <= t_max
            ]
        )
        return pi  # type: ignore

    def second_order_probability(self, key_a: KT, key_b: KT, t_max: int) -> float:
        dist_h = self.distributions
        min_part = 1.0 - np.product(
            [
                (
                    (1.0 - dist_h[t][key_a] - dist_h[t][key_b])
                    ** len(self.sampled_sets[t])
                )
                for t in dist_h
                if t <= t_max
            ]
        )
        pij = (
            self.inclusion_probability(key_a, t_max)
            + self.inclusion_probability(key_b, t_max)
            - min_part
        )
        return pij  # type: ignore

    def _sample(self, distribution: Sequence[Tuple[KT, float]]) -> Sequence[KT]:
        keys, probs = list_unzip(distribution)
        sample: Sequence[KT] = self.rng.choice(
            keys, size=self.batch_size, p=probs  # type: ignore
        ).tolist()
        return sample

    def _update_inclusion_prob(self, it: int):
        # Fixing some variables
        big_n = len(self.env.dataset)
        dists = self.distributions[it]
        prev_it = it - 1

        n = len(self.sampled_sets[it])
        dist = np.array([dists[k] for k in self.key_seq]).reshape((1, -1))
        if prev_it in self.cumulative_fo:
            prev_fo = self.cumulative_fo[prev_it]
            prev_so = self.cumulative_so[prev_it]
        else:
            prev_fo = np.zeros((1, big_n), dtype=self.dtype)
            prev_so = np.zeros((big_n, big_n), dtype=self.dtype)

        # Calculate first order (fo) part
        assert it not in self.cumulative_fo
        self.cumulative_fo[it] = prev_fo + n * np.log(1.0 - dist)

        # Calculate second order (so) part
        M = np.tile(dist, (big_n, 1))
        MT = M.T
        temp = 1.0 - M - MT
        # set diagonal values to 1 to make sure log calculation is valid
        np.fill_diagonal(temp, 1)

        assert it not in self.cumulative_so
        self.cumulative_so[it] = prev_so + n * np.log(temp)
        self.dist_fixed[it] = dist

        # Calculate fast binary label vector
        inclusions = frozenset(
            self.env.get_subset_by_labels(self.env.dataset, self.pos_label)
        )
        self.label_vector[it] = np.array([k in inclusions for k in self.key_seq])

    def fo_inclusion_probabilities(self, it: int) -> npt.NDArray[Any]:
        result = 1.0 - np.exp(self.cumulative_fo[it])
        return result

    def so_inclusion_probabilities(self, it: int) -> npt.NDArray[Any]:
        big_n = len(self.env.dataset)
        fo_prob = self.fo_inclusion_probabilities(it)
        M = np.tile(fo_prob, (big_n, 1))
        MT = M.T
        cu_so = self.cumulative_so[it]
        result = M + MT - (1.0 - np.exp(cu_so))
        return result

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)
            if self.it > 0:  # No previous iteration at the start
                # The previous sample has been finished
                # Update probabilities so we can estimate the number of inclusions
                prev_it = self.it - 1
                self.cumulative_sampled[prev_it] = frozenset(self.env.labeled)
                self._update_inclusion_prob(prev_it)
                if self.prune:
                    pprev_it = prev_it - 1
                    if pprev_it in self.cumulative_fo:
                        del self.cumulative_fo[pprev_it]
                        del self.cumulative_so[pprev_it]

            # Sample a new batch
            self._temp_augment_and_train()
            ranking = self._rank(self.env.dataset)
            distribution = calc_ap_prior_distribution(ranking)
            sample = self._sample(distribution)

            # Store all data for later analysis
            self.distributions[self.it] = dict(distribution)
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Calculate new sample size for the next iteration
            self.current_sample = collections.deque(sample)
            self.batch_size += ceil(self.batch_size / 10)
            self.it += 1

        return self.current_sample

    def __next__(self) -> IT:
        if len(self.env.unlabeled) != 1:
            return super().__next__()
        return next(iter(self.env.unlabeled.values()))
        