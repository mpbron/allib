from typing import Tuple

import imblearn # type: ignore
import numpy as np # type: ignore
from .base import BaseBalancer

class RandomOverSampler(BaseBalancer):
    """Balancing class that undersamples the data with a given ratio.
    """
    name = "RandomOverSampler"

    def __init__(self, random_state=0):
        super().__init__()
        self.ros = imblearn.over_sampling.RandomOverSampler(random_state=random_state)

    def resample(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_resampled, y_resampled = self.ros.fit_resample(x_data, y_data) # type: ignore
        return x_resampled, y_resampled # type: ignore