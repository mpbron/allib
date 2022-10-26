from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt

from ..environment import AbstractEnvironment

class BaseBalancer(ABC):
    """Abstract class for balance strategies."""
    _environment: Optional[AbstractEnvironment]
    name = "BaseBalancer"
    
    def __call__(self, environment: AbstractEnvironment) -> BaseBalancer:
        """Attach environment to Balancer. This
        can be used to add additional data to the training set
        
        Parameters
        ----------
        environment : AbstractEnvironment
            The environment used for this project
        
        Returns
        -------
        BaseBalancer
            The Balancer with the attached Environment
        """        
        self._environment = environment
        return self

    @abstractmethod
    def resample(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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
    def __init__(self) -> None:
        self._environment = None
        
    def resample(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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