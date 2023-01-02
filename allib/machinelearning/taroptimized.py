import itertools
import logging
from os import PathLike
from typing import (
    Any,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import instancelib as il
import math
import numpy as np
import numpy.typing as npt
from instancelib.labels.encoder import (
    DictionaryEncoder,
    IdentityEncoder,
    LabelEncoder,
    MultilabelDictionaryEncoder,
    SklearnLabelEncoder,
    SklearnMultiLabelEncoder,
)
from instancelib.exceptions.base import LabelEncodingException
from instancelib.machinelearning.sklearn import SkLearnClassifier
from instancelib.typehints.typevars import DT, KT, LMT, LT, PMT, RT, VT, DType
from instancelib.utils.func import list_unzip, zip_chain, filter_snd_none
from instancelib.utils.chunks import divide_iterable_in_lists
from instancelib.utils.saveablemodel import SaveableInnerModel
from instancelib.instances.vectorstorage import VectorStorage
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline  # type: ignore
from typing_extensions import Self

from sklearn.preprocessing import (
    LabelEncoder as SKLabelEncoder,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from ..balancing.base import BaseBalancer

from ..typehints.typevars import IT

LOGGER = logging.getLogger(__name__)


class ALSklearn(
    SkLearnClassifier[IT, KT, DT, VT, LT], Generic[IT, KT, DT, VT, LT, DType]
):
    vectorstorage: VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]]
    vectorizer: il.BaseVectorizer[IT]

    def __init__(
        self,
        estimator: Union[ClassifierMixin, Pipeline],
        encoder: LabelEncoder[LT, npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage: VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]],
        balancer: BaseBalancer,
        storage_location: "Optional[PathLike[str]]" = None,
        filename: "Optional[PathLike[str]]" = None,
        disable_tqdm: bool = True,
    ) -> None:
        super().__init__(
            estimator, encoder, storage_location, filename, disable_tqdm=disable_tqdm
        )
        self.vectorizer = vectorizer
        self.vectorstorage = vectorstorage
        self.balancer = balancer

    def encode_x(
        self, instances: Iterable[il.Instance[KT, Any, npt.NDArray[Any], Any]]
    ) -> Tuple[Sequence[KT], npt.NDArray[DType]]:
        ins_keys = [ins.identifier for ins in instances]
        or_keys, x_fm = self.vectorstorage.get_matrix(ins_keys)
        return or_keys, x_fm

    def encode_xy(
        self,
        instances: Iterable[il.Instance[KT, Any, npt.NDArray[Any], Any]],
        labelings: Iterable[Iterable[LT]],
    ) -> Tuple[Sequence[KT], npt.NDArray[DType], npt.NDArray[Any]]:
        ins_keys = [ins.identifier for ins in instances]
        lbl_dict = {
            ins.identifier: self.encoder.encode(lbl)
            for ins, lbl in zip(instances, labelings)
        }
        or_keys, x_fm = self.vectorstorage.get_matrix(ins_keys)
        y_lm = np.vstack([lbl_dict[k] for k in or_keys])
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return or_keys, x_fm, y_lm

    def _vectorize(self, provider: il.InstanceProvider[IT, KT, DT, VT, Any]) -> None:
        without_vectors = frozenset(provider).difference(self.vectorstorage)
        if without_vectors:
            keys = list(without_vectors)
            inss = [provider[key] for key in keys]
            vec_list = list(self.vectorizer.transform(inss))
            self.vectorstorage.add_bulk(keys, vec_list)

    def _get_preds(
        self, keys: Sequence[KT], matrix: npt.NDArray[DType]
    ) -> Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]:
        """Predict the labels for the current feature matrix

        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions

        Returns
        -------
        Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]
            A list of keys and the predictions belonging to it
        """
        pred_vec: npt.NDArray[Any] = self._predict(matrix)
        labels = self.encoder.decode_matrix(pred_vec)
        return keys, labels

    def _get_probas(
        self, keys: Sequence[KT], matrix: npt.NDArray[DType]
    ) -> Tuple[Sequence[KT], npt.NDArray[Any]]:
        """Calculate the probability matrix for the current feature matrix

        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[Any]]
            A list of keys and the probability predictions belonging to it
        """
        prob_vec: npt.NDArray[Any] = self._predict_proba(matrix)  # type: ignore
        return keys, prob_vec

    def _decode_proba_matrix(
        self, keys: Sequence[KT], y_matrix: npt.NDArray[Any]
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels))
        return zipped

    def predict_proba_provider_raw(
        self,
        provider: il.InstanceProvider[IT, KT, Any, npt.NDArray[Any], Any],
        batch_size: int = 200,
    ) -> Iterator[Tuple[Sequence[KT], npt.NDArray[Any]]]:
        matrices = self.vectorstorage.get_matrix_chunked(provider.key_list, batch_size)
        total_it = math.ceil(len(provider) / batch_size)
        preds = itertools.starmap(
            self._get_probas,
            tqdm(matrices, total=total_it, leave=False, disable=self._disable_tqdm),
        )
        yield from preds

    def predict_provider(
        self,
        provider: il.InstanceProvider[IT, KT, Any, npt.NDArray[Any], Any],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        matrices = self.vectorstorage.get_matrix_chunked(provider.key_list, batch_size)
        total_it = math.ceil(len(provider) / batch_size)
        preds = itertools.starmap(
            self._get_preds,
            tqdm(matrices, total=total_it, leave=False, disable=self._disable_tqdm),
        )
        results = list(zip_chain(preds))
        return results

    def fit_provider(
        self,
        provider: il.InstanceProvider[IT, KT, Any, npt.NDArray[Any], Any],
        labels: il.LabelProvider[KT, LT],
        batch_size: int = 200,
    ) -> None:
        LOGGER.info("[%s] Start with the fit procedure", self.name)
        # Collect the feature matrix for the labeled subset
        keys, matrix = self.vectorstorage.get_matrix(provider.key_list)
        LOGGER.info(
            "[%s] Gathered the feature matrix for all labeled documents",
            self.name,
        )

        # Get all labels for documents in the labeled set
        labelings = list(map(labels.get_labels, keys))
        LOGGER.info("[%s] Gathered all labels", self.name)
        LOGGER.info("[%s] Start fitting the classifier", self.name)
        self._fit_vectors(matrix, labelings)
        LOGGER.info("[%s] Fitted the classifier", self.name)

    def _fit_vectors(
        self,
        x_data: npt.NDArray[DType],
        labels: Sequence[FrozenSet[LT]],
    ):
        x_mat, y_mat = self._filter_x_only_encoded_y(x_data, labels)
        self._fit(x_mat, y_mat)

    def _fit(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]):
        x_resampled, y_resampled = self.balancer.resample(x_data, y_data)
        return super()._fit(x_resampled, y_resampled)

    def _filter_x_only_encoded_y(
        self, matrix: npt.NDArray[DType], labelings: Sequence[Iterable[LT]]
    ) -> Tuple[npt.NDArray[DType], npt.NDArray[Any]]:
        """Filter out the training data for which no label exists

        Parameters
        ----------
        instances : Iterable[_T]
            Training instances
        labelings : Sequence[Iterable[LT]]
            The labels

        Returns
        -------
        Tuple[Iterable[_T], npt.NDArray[Any]]
            A tuple containing the training instances and a label matrix that contains all succesfully encoded labels
        """
        try:
            y_mat = self.encoder.encode_batch(labelings)
        except LabelEncodingException:
            idxs = range(matrix.shape[0])
            y_vecs = map(self.encoder.encode_safe, labelings)
            lbl_idx, lbls = filter_snd_none(idxs, y_vecs)
            lbl_idx_set = frozenset(lbl_idx)
            x_mask = [idx in lbl_idx_set for idx in idxs]
            x_mat = matrix[x_mask, :]
            y_mat = np.vstack(lbls)
        else:
            x_mat = matrix
        return x_mat, y_mat

    @staticmethod
    def vectorize(
        env: il.Environment[IT, KT, DT, VT, Any, LT],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage: VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]],
        chunk_size: int = 2000,
    ) -> VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]]:
        provider = env.all_instances
        instances = list(
            itertools.chain.from_iterable(provider.instance_chunker(chunk_size))
        )
        vectorizer.fit(instances)
        matrix = vectorizer.transform(instances)
        total_it = math.ceil(len(instances) / chunk_size)
        chunks = divide_iterable_in_lists(instances, chunk_size)
        for instance_chunk in tqdm(chunks, total=total_it):
            keys = [ins.identifier for ins in instance_chunk]
            matrix = vectorizer.transform(instance_chunk)
            vectorstorage.add_bulk_matrix(keys, matrix)
        return vectorstorage

    @classmethod
    def build(
        cls,
        estimator: Union[ClassifierMixin, Pipeline],
        env: il.Environment[IT, KT, DT, VT, Any, LT],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage: VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]],
        balancer: BaseBalancer,
        chunk_size: int = 2000,
        storage_location: "Optional[PathLike[str]]" = None,
        filename: "Optional[PathLike[str]]" = None,
    ) -> SkLearnClassifier[IT, KT, DT, VT, LT]:
        """Construct a Sklearn model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`)
        env : Environment[IT, KT, Any, npt.NDArray[Any], Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        sklearn_encoder: TransformerMixin = SKLabelEncoder()
        il_encoder = SklearnLabelEncoder(sklearn_encoder, env.labels.labelset)
        vectorstorage = cls.vectorize(env, vectorizer, vectorstorage, chunk_size)
        return cls(
            estimator,
            il_encoder,
            vectorizer,
            vectorstorage,
            balancer,
            storage_location,
            filename,
        )

    @classmethod
    def build_multilabel(
        cls,
        estimator: Union[ClassifierMixin, Pipeline],
        env: il.Environment[IT, KT, Any, npt.NDArray[Any], Any, LT],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage: VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]],
        balancer: BaseBalancer,
        chunk_size: int = 2000,
        storage_location: "Optional[PathLike[str]]" = None,
        filename: "Optional[PathLike[str]]" = None,
    ) -> Self:
        """Construct a Sklearn model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
             The scikit-learn API Classifier capable of Multilabel Classification
        env : Environment[IT, KT, Any, npt.NDArray[Any], Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, VT, LT]:
            The model
        """
        sklearn_encoder: TransformerMixin = MultiLabelBinarizer()
        il_encoder = SklearnMultiLabelEncoder(sklearn_encoder, env.labels.labelset)
        vectorstorage = cls.vectorize(env, vectorizer, vectorstorage, chunk_size)
        return cls(
            estimator,
            il_encoder,
            vectorizer,
            vectorstorage,
            balancer,
            storage_location,
            filename,
        )
