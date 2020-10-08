# Copyright 2019 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import log, floor
from typing import Tuple, Optional

import numpy as np # type: ignore

from .base import BaseBalancer
from .base import IdentityBalancer
from ..utils import get_random_generator

class DoubleBalancer(BaseBalancer):
    """Dynamic Resampling balance strategy.

    Class to get the two way rebalancing function and arguments.
    It super samples ones depending on the number of 0's and total number
    of samples in the training data.

    Arguments
    ---------
    a: float
        Governs the weight of the 1's. Higher values mean linearly more 1's
        in your training sample.
    alpha: float
        Governs the scaling the weight of the 1's, as a function of the
        ratio of ones to zeros. A positive value means that the lower the
        ratio of zeros to ones, the higher the weight of the ones.
    b: float
        Governs how strongly we want to sample depending on the total
        number of samples. A value of 1 means no dependence on the total
        number of samples, while lower values mean increasingly stronger
        dependence on the number of samples.
    beta: float
        Governs the scaling of the weight of the zeros depending on the
        number of samples. Higher values means that larger samples are more
        strongly penalizing zeros.
    """

    name = "DoubleBalancer"

    def __init__(self, a=2.155, alpha=0.94, b=0.789, beta=1.0,
                 rng:  Optional[np.random.Generator] = None): # type: ignore
        super(DoubleBalancer, self).__init__()
        self.a = a
        self.alpha = alpha
        self.b = b
        self.beta = beta
        self.fallback_model = IdentityBalancer()
        self._rng = get_random_generator(rng) # type: ignore

    def resample(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if y_data.ndim > 1:
            positive_indices = np.where(np.any(y_data == 1, axis=1))[0] # type: ignore
            negative_indices = np.where(np.all(y_data == 0, axis=1))[0]
        else:
            positive_indices = np.where(y_data == 1)[0]
            negative_indices = np.where(y_data == 0)[0]

        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        n_train = n_positive + n_negative

        # Fall back to simple sampling if we have only ones or zeroes.
        if n_positive == 0 or n_negative == 0:
            return x_data, y_data

        # Compute sampling weights.
        one_weight = _one_weight(n_positive, n_negative, self.a, self.alpha)
        zero_weight = _zero_weight(n_positive + n_negative, self.b, self.beta)
        tot_zo_weight = one_weight * n_positive + zero_weight * n_negative
        # Number of inclusions to sample.
        n_positive_train = random_round(
            one_weight * n_positive * n_train / tot_zo_weight, self._rng)
        # Should be at least 1, and at least two spots should be for exclusions.
        n_positive_train = max(1, min(n_train - 2, n_positive_train))
        # Number of exclusions to sample
        n_negative_train = n_train - n_positive_train

        # Sample records of ones and zeroes
        pos_train_idx = fill_training(
            positive_indices, n_positive_train, self._rng)
        neg_train_idx = fill_training(negative_indices, n_negative_train,
                                      self._rng)
        # Merge and shuffle.
        all_idx = np.concatenate([pos_train_idx, neg_train_idx])
        shuf_idx = self._rng.permutation(all_idx)

        # Return resampled feature matrix and labels.
        return x_data[shuf_idx], y_data[shuf_idx]


def _one_weight(n_one: int, n_zero: int, a: float, alpha: float) -> float:
    """Get the weight of the ones."""
    weight = a * (n_one / n_zero)**(-alpha)
    return weight


def _zero_weight(n_read: int, b: float, beta: float) -> float:
    """Get the weight of the zeros."""
    weight = 1 - (1 - b) * (1 + log(n_read))**(-beta)
    return weight


def random_round(value: float, rng: np.random.Generator) -> int: # type: ignore
    """Round up or down, depending on how far the value is.

    For example: 8.1 would be rounded to 8, 90% of the time, and rounded
    to 9, 10% of the time.
    """
    base = int(floor(value))
    if rng.random() < value - base:
        base += 1
    return base


def fill_training(src_idx: np.ndarray, n_train: int, rng: np.random.Generator) -> np.ndarray: # type: ignore
    """Copy/sample until there are n_train indices sampled/copied.
    """
    # Number of copies needed.
    n_copy = np.int(n_train / len(src_idx)) # type: ignore
    # For the remainder, use sampling.
    n_sample = n_train - n_copy * len(src_idx)

    # Copy indices
    dest_idx = np.tile(src_idx, n_copy).reshape(-1)
    # Add samples
    dest_idx = np.append(dest_idx,
                         rng.choice(src_idx, n_sample, replace=False))
    return dest_idx
