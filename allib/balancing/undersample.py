
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
from typing import Optional

from math import ceil

import numpy as np # type: ignore

from .base import BaseBalancer
from ..utils import get_random_generator

class UndersampleBalancer(BaseBalancer):
    """Balancing class that undersamples the data with a given ratio.
    """
    name = "Undersample"

    def __init__(self, ratio=1.0, rng: Optional[np.random.Generator] = None):
        """Initialize the undersampling balance strategy.
        Arguments
        ---------
        ratio: double
            Undersampling ratio of the zero's. If for example we set a ratio of
            0.25, we would sample only a quarter of the zeros and all the ones.
        """
        self.ratio = ratio
        self._rng = get_random_generator(rng)

    def resample(self, x_data, y_data):
        # Find all documents where that has a label:
        # This assumes that negative examples are 
        # encoded as zeroes. Works for multiclass,
        # multilabel. Works for binary classification
        # if the positive class is encoded as 1.
        if y_data.ndim > 1:
            positive_indices = np.where(np.any(y_data == 1, axis=1))[0]
            negative_indices = np.where(np.all(y_data == 0, axis=1))[0]
        else:
            positive_indices = np.where(y_data == 1)[0]
            negative_indices = np.where(y_data == 0)[0]

        n_positive = len(positive_indices)
        n_negative = len(negative_indices)

        # Calculate the ratio between positive and 
        # negative
        # If we don't have an excess of negative examples,
        # give back all training_samples.
        if n_positive / n_negative >= self.ratio:
            sample_ind = np.append(
                positive_indices, negative_indices)
        else:
            n_negative_sample = ceil(n_positive / self.ratio)
            negative_sample = self._rng.choice(
                np.arange(n_negative), n_negative_sample, replace=False)
            sample_ind = np.append(positive_indices, 
            negative_indices[negative_sample])
        shuf_ind = self._rng.permutation(sample_ind)
        return x_data[shuf_ind], y_data[shuf_ind]

    