from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from typing import Any


class AbstractSelectionCriterion(ABC):
    name: str = "AbstractSelectionCriterion"

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def __call__(self, prob_mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculates the selection metric given a probability matrix

        Parameters
        ----------
        prob_mat : npt.NDArray[Any]
            The probability matrix with rows of class probabilities. 
            Shape should be ``(n_instances, n_classes)``

        Returns
        -------
        npt.NDArray[Any]
            The result of the selection metrix. This has as shape
            ``(n_instances, )``

        """
        raise NotImplementedError