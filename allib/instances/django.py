from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from uuid import uuid4
import itertools
from itertools import chain
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
from django.db.models import QuerySet

from instances.base import (ChildInstance, Instance, InstanceProvider,
                            ParentInstance, Matrix)
from teacher_api.django_queries import (get_set_documents_pks,
                                        get_set_documents_vectors)
from teacher_api.models import (Document, DocumentSet, FragmentLevel, Project,
                                TextFragment, User, Dataset)

from teacher_api.utils import decode_vector, is_default_b64


def get_or_create_root() -> User:
    try:
        try:
            return User.objects.get(username="MT_ROOT")
        except User.DoesNotExist:
            uuid = uuid4()
            email = "{}@machineteacher.intern".format(uuid)
            user = User.objects.create_user(
                username="MT_ROOT", password=uuid, email=email)
            return user
    except:
        return None


ROOT_USER = get_or_create_root()


class NamedProvider(Enum):
    WHOLE_DATASET = "WholeDataset"
    UNLABELED = "Unlabeled"
    LABELED = "Labeled"


class DjangoInstance(Instance):
    def __init__(self, identifier, text, vector) -> None:
        self._identifier = identifier
        self._data = text
        self._vector = vector

    @property
    def data(self) -> str:
        return self._data

    @property
    def representation(self) -> str:
        return self._data

    @property
    def identifier(self) -> int:
        return self._identifier

    @property
    def vector(self) -> Optional[np.ndarray]:
        if self._vector is None:
            doc: Document = Document.objects.get(pk=self._identifier)
            self._vector = doc.vector
        return self._vector

    @vector.setter
    def vector(self, value):
        doc = Document.objects.get(pk=self._identifier)
        doc.vector = value
        doc.save()
        self._vector = value

    @classmethod
    def from_document(cls, doc: Document):
        return cls(
            doc.pk,
            str(doc.text),
            doc.vector)


class DjangoSequenceInstance(DjangoInstance, ParentInstance):
    def __init__(self, identifier, text, vector) -> None:
        super().__init__(identifier, text, vector)
        fragments = Document.objects.get(pk=identifier).fragments.all()
        self.children = [DjangoFragInstance.from_fragment(
            self, frag) for frag in fragments]

    @property
    def children(self) -> List[ChildInstance]:
        return self.children


class DjangoFragInstance(ChildInstance):
    start: int
    end: int
    level: FragmentLevel

    def __init__(self,
                 identifier, parent: DjangoSequenceInstance,
                 level: FragmentLevel,
                 start: int, end: int,
                 vector):
        self._identifier = identifier
        self._parent = parent
        self.start = start
        self.end = end
        self._vector = vector
        self.level = level

    @property
    def data(self) -> str:
        return self._parent.data[self.start:self.end]

    @property
    def representation(self) -> str:
        return self.data

    @property
    def vector(self) -> Optional[np.ndarray]:
        return self._vector

    @property
    def identifier(self) -> str:
        return self._identifier

    @vector.setter
    def vector(self, value) -> None:
        fragment = TextFragment.objects.get(pk=self._identifier)
        fragment.vector = value
        self._vector = value

    @property
    def parent(self) -> DjangoSequenceInstance:
        return self._parent

    @classmethod
    def from_fragment(cls, parent: DjangoSequenceInstance, frag: TextFragment):
        return cls(
            frag.pk,
            parent,
            frag.level,
            frag.start, frag.end,
            frag.get_vector()
        )


class DjangoProvider(InstanceProvider, ABC):
    def _build_instance(self, key: int) -> DjangoInstance:
        doc = Document.objects.get(pk=key)
        return self._build_instance_from_doc(doc)

    def _build_instance_from_doc(self, doc: Document) -> DjangoInstance:
        return DjangoInstance.from_document(doc)

    @property
    @abstractmethod
    def _documents(self) -> QuerySet:
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return chain.from_iterable(self._documents.all().values_list("pk"))

    @property
    def key_list(self):
        return list(chain.from_iterable(self._documents.all().values_list("pk")))

    def __getitem__(self, key: int) -> Instance:
        if self.__contains__(key):
            return self._build_instance(key)
        raise KeyError(f"The document with key \"{key}\" is not present in this Provider")

    def add(self, instance: DjangoInstance) -> None:
        try:
            self.__setitem__(instance.identifier, instance)
        except Document.DoesNotExist:
            # This should never happen, this means database corruption
            raise KeyError("The document with key \"{instance.identifier}\" is not present in the Django Database".format(
                instance.identifier))

    def discard(self, instance: DjangoInstance) -> None:
        try:
            self.__delitem__(instance.identifier)
        except KeyError:
            # To adhere to the behavior of Set.discard() which does not raise a KeyError if it is not present in the set.
            pass

    def __len__(self) -> int:
        return self._documents.count()

    @property
    def empty(self) -> bool:
        return self._documents.first() is None

    def get_all(self) -> Iterator[DjangoInstance]:
        return (self._build_instance_from_doc(doc) for doc in self._documents.all())

    def __contains__(self, key: int) -> bool:
        return self._documents.filter(pk=key).exists()

    @property
    def feature_matrix(self) -> Optional[Matrix]:
        if self._feature_matrix is None:
            pk_vector_tuple_list = list(
                self._documents.all().values_list('pk', '_vector'))
            decoded = [(pk, decode_vector(b64s)) for (pk, b64s) in pk_vector_tuple_list if not is_default_b64(b64s)]
            if decoded:
                pks, vectors = zip(*decoded)
                self._feature_matrix = Matrix(np.vstack(vectors), list(pks))
        return self._feature_matrix

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = {key: value for (key, value) in self.__dict__.items() if key != "_feature_matrix"}
        # Remove the unpicklable entries.
        state = {**state.copy(), **{"_feature_matrix": None}}
        return state



class DjangoBucketProvider(DjangoProvider):
    def __init__(self, project: Project, identifier: str, user: User):
        self.project = project
        self.identifier = identifier
        self.user = user
        self._feature_matrix = None
    
    @property
    def _documents(self) -> QuerySet:
        return self._docset.set_docs

    @property
    def _docset(self) -> DocumentSet:
        docset, _ = DocumentSet.objects.get_or_create(project=self.project, name=self.identifier)
        return docset

    def __setitem__(self, key: int, value: DjangoInstance) -> None:
        if self.__contains__(key):
            return
        try:
            self._docset.add(key)
        except Document.DoesNotExist:
            raise KeyError(
                "The document with key \"{}\" is not present in the Django Database".format(key))

    def __delitem__(self, key: int) -> None:
        if self.__contains__(key):
            self._docset.remove(key)
        raise KeyError(
            "The document with key \"{}\" is not present in this Provider".format(key))

    def clear(self) -> None:
        self._docset.clear()


class DjangoDatasetProvider(DjangoProvider):
    def __init__(self, project: Project):
        self.project = project
        self._feature_matrix = None

    @property
    def _documents(self) -> QuerySet:
        # TODO check if this query is efficient
        datasets = list(Dataset.objects.filter(project=self.project).all())
        return Document.objects.filter(dataset__in=datasets)

    def __setitem__(self, key: int, value: DjangoInstance) -> None:
        pass

    def __delitem__(self, key: int) -> None:
        pass

    def add(self, instance: DjangoInstance) -> None:
        pass

    def discard(self, instance: DjangoInstance) -> None:
        pass

    def clear(self) -> None:
        pass


class DjangoUnlabeledProvider(DjangoDatasetProvider):
    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.labeled = DjangoBucketProvider(
            project, NamedProvider.LABELED, ROOT_USER)

    def __contains__(self, key: int) -> bool:
        return super().__contains__(key) and not self.labeled.__contains__(key)

    @property
    def _documents(self) -> QuerySet:
        return super()._documents.exclude(pk__in=self.labeled.key_list).filter(sampleble=True)

