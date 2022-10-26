from distutils.command.build import build
from os import PathLike
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from instancelib import BaseVectorizer, Environment
from instancelib.labels.encoder import LabelEncoder, SklearnLabelEncoder
from instancelib.machinelearning.autovectorizer import AutoVectorizerClassifier
from instancelib.machinelearning.skdata import SkLearnDataClassifier
from instancelib.machinelearning.sklearn import SkLearnClassifier

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import \
    LabelEncoder as SKLabelEncoder  # type: ignore

from ..balancing import BaseBalancer
from ..typehints import DT, IT, KT, LT


class BalancedSklearnVectorClassifier(AutoVectorizerClassifier[IT, KT, LT]):
    def __init__(self, 
                 estimator: Union[ClassifierMixin, Pipeline], 
                 vectorizer: BaseVectorizer[IT], 
                 encoder: LabelEncoder[LT, npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
                 balancer: BaseBalancer, 
                 storage_location: "Optional[PathLike[str]]" = None, 
                 filename: "Optional[PathLike[str]]" = None) -> None:
        super().__init__(estimator, vectorizer, encoder, storage_location, filename)
        self.balancer = balancer

    def _fit(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]):
        x_resampled, y_resampled = self.balancer.resample(x_data, y_data)
        return super()._fit(x_resampled, y_resampled)

    @classmethod
    def build(cls, estimator: Union[ClassifierMixin, Pipeline],
                   vectorizer: BaseVectorizer[IT],
                   balancer: BaseBalancer,  
                   env: Environment[IT, KT, Any, npt.NDArray[Any], Any, LT], 
                   storage_location: "Optional[PathLike[str]]" = None, 
                   filename: "Optional[PathLike[str]]" = None
                   ) -> SkLearnClassifier[IT, KT, Any, npt.NDArray[Any], LT]:
        sklearn_encoder: TransformerMixin = SKLabelEncoder()
        il_encoder = SklearnLabelEncoder(sklearn_encoder, env.labels.labelset)    
        return cls(estimator, vectorizer, il_encoder, balancer, storage_location, filename)


class BalancedDataClassifier(SkLearnDataClassifier[IT, KT, DT, LT]):
    def __init__(self, 
                 estimator: Union[ClassifierMixin, Pipeline], 
                 encoder: LabelEncoder[LT, npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
                 balancer: BaseBalancer,
                 storage_location: "Optional[PathLike[str]]" = None, 
                 filename: "Optional[PathLike[str]]" = None) -> None:
        super().__init__(estimator, encoder, storage_location, filename)
        self.balancer = balancer

    def _fit(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]):
        x_resampled, y_resampled = self.balancer.resample(x_data, y_data)
        return super()._fit(x_resampled, y_resampled)

    @classmethod
    def build(cls, estimator: Union[ClassifierMixin, Pipeline],
                   balancer: BaseBalancer,  
                   env: Environment[IT, KT, Any, npt.NDArray[Any], Any, LT], 
                   storage_location: "Optional[PathLike[str]]" = None, 
                   filename: "Optional[PathLike[str]]" = None
                   ) -> SkLearnClassifier[IT, KT, Any, npt.NDArray[Any], LT]:
        sklearn_encoder: TransformerMixin = SKLabelEncoder()
        il_encoder = SklearnLabelEncoder(sklearn_encoder, env.labels.labelset)    
        return cls(estimator, il_encoder, balancer, storage_location, filename)
