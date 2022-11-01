from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    MutableMapping,
    Sequence,
    Union,
)
from uuid import UUID

import instancelib as il

from instancelib import InstanceProvider
from instancelib.instances.memory import (
    DataPoint,
    DataPointProvider,
    MemoryBucketProvider,
)
from instancelib.labels.base import LabelProvider
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.typehints import DT, KT, LT, RT, VT
from instancelib.utils.func import union

from ..history import MemoryLogger
from ..history.base import BaseLogger
from .base import IT, AbstractEnvironment

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)


class AbstractMemoryEnvironment(
    AbstractEnvironment[IT, KT, DT, VT, RT, LT], ABC, Generic[IT, KT, DT, VT, RT, LT]
):

    _public_dataset: InstanceProvider[IT, KT, DT, VT, RT]
    _dataset: InstanceProvider[IT, KT, DT, VT, RT]
    _unlabeled: InstanceProvider[IT, KT, DT, VT, RT]
    _labeled: InstanceProvider[IT, KT, DT, VT, RT]
    _labelprovider: LabelProvider[KT, LT]
    _truth: LabelProvider[KT, LT]
    _logger: BaseLogger[KT, LT, Any]
    _named_providers: MutableMapping[str, InstanceProvider[IT, KT, DT, VT, RT]]

    def __contains__(self, __o: object) -> bool:
        return __o in self._named_providers

    def __getitem__(self, __k: str) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self._named_providers[__k]

    def __setitem__(self, __k: str, __v: InstanceProvider[IT, KT, DT, VT, RT]) -> None:
        self.set_named_provider(__k, __v)

    def __len__(self) -> int:
        return len(self._named_providers)

    def __delitem__(self, __v: str) -> None:
        del self._named_providers[__v]

    def __iter__(self) -> Iterator[str]:
        return iter(self._named_providers)

    @property
    def dataset(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self._public_dataset

    @property
    def all_instances(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self._dataset

    @property
    def labels(self) -> LabelProvider[KT, LT]:
        return self._labelprovider

    @property
    def logger(self) -> BaseLogger[KT, LT, Any]:
        return self._logger

    @property
    def unlabeled(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self._unlabeled

    @property
    def labeled(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self._labeled

    @property
    def truth(self) -> LabelProvider[KT, LT]:
        return self._truth

    def create_bucket(self, keys: Iterable[KT]) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return MemoryBucketProvider[IT, KT, DT, VT, RT](self._dataset, keys)

    def create_empty_provider(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
        return self.create_bucket([])

    def set_named_provider(
        self, name: str, value: InstanceProvider[IT, KT, DT, VT, RT]
    ):
        self._named_providers[name] = value

    def create_named_provider(
        self, name: str, keys: Iterable[KT] = list()
    ) -> InstanceProvider[IT, KT, DT, VT, RT]:
        self._named_providers[name] = self.create_bucket(keys)
        return self._named_providers[name]


class MemoryEnvironment(
    AbstractMemoryEnvironment[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def __init__(
        self,
        dataset: InstanceProvider[IT, KT, DT, VT, RT],
        unlabeled: InstanceProvider[IT, KT, DT, VT, RT],
        labeled: InstanceProvider[IT, KT, DT, VT, RT],
        named_providers: MutableMapping[str, InstanceProvider[IT, KT, DT, VT, RT]],
        public_dataset: InstanceProvider[IT, KT, DT, VT, RT],
        labelprovider: LabelProvider[KT, LT],
        logger: BaseLogger[KT, LT, Any],
        truth: LabelProvider[KT, LT],
    ):
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._named_providers = named_providers
        self._logger = logger
        self._truth = truth
        self._public_dataset = public_dataset

    @classmethod
    def from_environment(
        cls,
        environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        shared_labels: bool = True,
        *args,
        **kwargs,
    ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        dataset = environment.all_instances
        unlabeled = MemoryBucketProvider(dataset, environment.unlabeled.key_list)
        labeled = MemoryBucketProvider(dataset, environment.labeled.key_list)
        if shared_labels:
            labels = environment.labels
        else:
            labels = MemoryLabelProvider[KT, LT].from_data(
                environment.labels.labelset, [], []
            )
        logger = environment.logger
        truth = environment.truth
        named_providers: Dict[str, InstanceProvider[IT, KT, DT, VT, RT]] = {
            key: MemoryBucketProvider(dataset, prov.key_list)
            for key, prov in environment.items()
        }
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )

    @classmethod
    def from_environment_only_data(
        cls, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        dataset = environment.all_instances
        unlabeled = MemoryBucketProvider(dataset, dataset.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[KT, LT](environment.labels.labelset, {}, {})
        logger = MemoryLogger[KT, LT, Any](labels)
        truth = environment.truth
        named_providers: Dict[str, InstanceProvider[IT, KT, DT, VT, RT]] = {
            key: MemoryBucketProvider(dataset, prov.key_list)
            for key, prov in environment.items()
        }
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )

    @classmethod
    def from_instancelib(
        cls, environment: il.AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        dataset = environment.all_instances
        labeled_docs = union(
            *(
                environment.labels.get_instances_by_label(label)
                for label in environment.labels.labelset
            )
        )
        unlabeled_docs = frozenset(dataset.key_list).difference(labeled_docs)
        unlabeled = MemoryBucketProvider(dataset, unlabeled_docs)
        labeled = MemoryBucketProvider(dataset, labeled_docs)
        labels = MemoryLabelProvider[KT, LT].from_provider(environment.labels)
        environment.labels
        logger = MemoryLogger[KT, LT, Any](labels)
        truth = MemoryLabelProvider[KT, LT].from_provider(environment.labels)
        named_providers: Dict[str, InstanceProvider[IT, KT, DT, VT, RT]] = {
            key: MemoryBucketProvider(dataset, prov.key_list)
            for key, prov in environment.items()
        }
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )

    @classmethod
    def from_instancelib_simulation(
        cls, environment: il.AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        dataset = environment.all_instances
        unlabeled = MemoryBucketProvider(dataset, dataset.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[KT, LT].from_data(
            environment.labels.labelset, [], []
        )
        logger = MemoryLogger[KT, LT, Any](labels)
        truth = MemoryLabelProvider[KT, LT].from_provider(environment.labels)
        named_providers: Dict[str, InstanceProvider[IT, KT, DT, VT, RT]] = {
            key: MemoryBucketProvider(dataset, prov.key_list)
            for key, prov in environment.items()
        }
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )

    @classmethod
    def from_instancelib_simulation_heldout(
        cls,
        environment: il.AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        train_set: il.InstanceProvider[IT, KT, DT, VT, RT],
    ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
        dataset = environment.all_instances
        unlabeled = MemoryBucketProvider(dataset, train_set.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[KT, LT].from_data(
            environment.labels.labelset, [], []
        )
        logger = MemoryLogger[KT, LT, Any](labels)
        truth = MemoryLabelProvider[KT, LT].from_provider(environment.labels)
        named_providers: Dict[str, InstanceProvider[IT, KT, DT, VT, RT]] = {
            key: MemoryBucketProvider(dataset, prov.key_list)
            for key, prov in environment.items()
        }
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )


class DataPointEnvironment(
    MemoryEnvironment[
        DataPoint[Union[KT, UUID], DT, VT, RT], Union[KT, UUID], DT, VT, RT, LT
    ],
    Generic[KT, DT, VT, RT, LT],
):
    @classmethod
    def from_data(
        cls,
        target_labels: Iterable[LT],
        indices: Sequence[KT],
        data: Sequence[DT],
        ground_truth: Sequence[Iterable[LT]],
        vectors: Sequence[VT],
    ) -> DataPointEnvironment[KT, DT, VT, RT, LT]:
        dataset = DataPointProvider[KT, DT, VT, RT].from_data_and_indices(
            indices, data, vectors
        )
        unlabeled = MemoryBucketProvider(dataset, dataset.key_list)
        labeled = MemoryBucketProvider(dataset, [])
        labels = MemoryLabelProvider[Union[KT, UUID], LT].from_data(
            target_labels, indices, []
        )
        logger = MemoryLogger[Union[KT, UUID], LT, Any](labels)
        truth = MemoryLabelProvider[Union[KT, UUID], LT].from_data(
            target_labels, indices, ground_truth
        )
        named_providers = dict()
        public_dataset = MemoryBucketProvider(dataset, dataset.key_list)
        return cls(
            dataset,
            unlabeled,
            labeled,
            named_providers,
            public_dataset,
            labels,
            logger,
            truth,
        )
