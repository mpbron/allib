from abc import ABC, abstractmethod

import numpy as np


class AbstractSelectionCriterion(ABC):
    name: str = "AbstractSelectionCriterion"

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        """Calculates the selection metric given a probability matrix

        Parameters
        ----------
        prob_mat : np.ndarray
            The probability matrix with rows of class probabilities. 
            Shape should be ``(n_instances, n_classes)``

        Returns
        -------
        np.ndarray
            The result of the selection metrix. This has as shape
            ``(n_instances, )``

        """
        raise NotImplementedError