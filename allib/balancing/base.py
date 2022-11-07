from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Tuple, Any

import numpy.typing as npt


class BaseBalancer(ABC):
    """Abstract class for balance strategies."""

    name = "BaseBalancer"

    @abstractmethod
    def resample(
        self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Resample the training data

        Parameters
        ----------
        x_data : npt.NDArray[Any]
            The feature matrix of the training data
        y_data : npt.NDArray[Any]
            The encoded labels for all the training data

        Returns
        -------
        Tuple[npt.NDArray[Any], npt.NDArray[Any]]
            the resampled feature matrix, and corresponding labels.


        """
        raise NotImplementedError


class IdentityBalancer(BaseBalancer):
    def resample(
        self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Resample the training data (identity function)

        Parameters
        ----------
        x_data : npt.NDArray[Any]
            The feature matrix of the training data
        y_data : npt.NDArray[Any]
            The encoded labels for all the training data

        Returns
        -------
        Tuple[npt.NDArray[Any], npt.NDArray[Any]]
            the resampled feature matrix, and corresponding labels.
        """
        return x_data, y_data
