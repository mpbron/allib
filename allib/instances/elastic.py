from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
from django.db.models import QuerySet
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from backend.ext_connections import get_es_project_init

from instances.django import (DjangoBucketProvider, DjangoDatasetProvider,
                              DjangoInstance, DjangoProvider,
                              DjangoUnlabeledProvider,
                              ROOT_USER, NamedProvider)

from teacher_api.models import Document, Project, User


def es_doc_query(identifier, identifier_fieldname) -> Dict[str, str]:
    """Function that creates a query to retrieve the text 
    field from an Elastic object for a given identifier. 
    This is the default query for viewing an Elastic 
    document in the frontend

    Parameters
    ----------
    identifier : str
        the identifier of a document within an Elastic Search instance

    Returns
    -------
    Dict[str,str]
        The query that can be used to retrieve the document
    """
    query_doc_text = {
        "_source": ["text"],
        "query": {
            "bool": {
                "must": {"match": {identifier_fieldname: identifier}}
            }
        }
    }
    return query_doc_text


def es_query_terms(query_string, already_labeled):
    labeled_identifiers = [{"match": {"identifier": identifier}}
                           for identifier in set(already_labeled)]
    query_multiple_terms = {
        "_source": ["identifier", "text"],
        "track_scores": "true",
        "min_score": 2.5,
        "query": {
            "bool": {
                "must_not": labeled_identifiers,
                "should": {
                    "match": {
                        "text": {
                            "query": query_string,
                            "fuzziness": "AUTO"
                        }
                    }
                }
            }
        }
    }
    return query_multiple_terms


class ElasticConnection:
    def __init__(self, project: Project):
        self.identifier_field_name = project.elastic_identifier_field
        self.project = project

    @property
    def connection(self) -> Elasticsearch:
        connection, _ = get_es_project_init(self.project.name)
        return connection

    @property
    def index(self) -> str:
        _, index = get_es_project_init(self.project.name)
        return index
    
    def get_es_text(self, identifier) -> str:
        es_results = scan(
            self.connection,
            query=es_doc_query(identifier, self.identifier_field_name),
            index=self.index,
            size=1000)
        es_data: pd.DataFrame = pd.json_normalize(
            [hit['_source'] for hit in list(es_results)])
        if es_data.empty:
            raise KeyError("Document not found in Elastic")
        doc_text = es_data.text.iloc[0]
        return doc_text
    
    def search_query(self, terms: str, exclude_es_identifiers: List[str], size: int) -> List[str]:
        es_results = self.connection.search(
            body=es_query_terms(terms, exclude_es_identifiers),
            index=self.index, size=size)
        es_data = pd.json_normalize([hit['_source']
                                  for hit in list(es_results['hits']['hits'])])
        identifiers = list(es_data.loc[:, 'identifier'])
        return identifiers
    

class ElasticInstance(DjangoInstance):
    @classmethod
    def from_doc_and_es(cls, doc: Document, text: str):
        return cls(
            doc.pk,
            text,
            doc.vector)


class ElasticLazy(DjangoInstance):
    def __init__(self,
                 esc: ElasticConnection,
                 identifier, elastic_identifier, data=None, vector=None):
        super().__init__(identifier, data, vector)
        self._esc = esc
        self._identifier = identifier
        self.elastic_identifier = elastic_identifier

    @property
    def data(self) -> str:
        if self._data is None:
            self._data = self._esc.get_es_text(self.elastic_identifier)
        return self._data


class ElasticProvider(DjangoProvider, ABC):
    @abstractmethod
    def __init__(self, project: Project, esc: ElasticConnection) -> None:
        self.project = project
        self._esc = esc
        self._feature_matrix = None

    def _build_instance(self, key: int) -> DjangoInstance:
        doc = Document.objects.get(pk=key)
        return self._build_instance_from_doc(doc)

    def _build_instance_from_doc(self, doc: Document) -> ElasticInstance:
        return ElasticLazy(self._esc, doc.pk, doc.identifier)

    def search_query(self, terms: str, exclude_list: List[int], size: int) -> List[ElasticInstance]:
        exclude_es_identifiers = list(self._documents.filter(
            pk__in=exclude_list).values_list("identifier", flat=True))
        es_identifiers = self._esc.search_query(terms, exclude_es_identifiers, size)
        docs = self._documents.filter(identifier__in=es_identifiers).all()
        return [self._build_instance_from_doc(doc) for doc in docs]
    

class ElasticBucketProvider(ElasticProvider, DjangoBucketProvider):
    def __init__(self,
                 project: Project,
                 identifier: str,
                 user: User,
                 esc: ElasticConnection) -> None:
        super(ElasticBucketProvider, self).__init__(project, esc)
        self.identifier = identifier
        self.user = user
        self._feature_matrix = None


class ElasticDatasetProvider(ElasticProvider, DjangoDatasetProvider):
    def __init__(self,
                 project: Project,
                 esc: ElasticConnection) -> None:
        super(ElasticDatasetProvider, self).__init__(project, esc)


class ElasticUnlabeledProvider(ElasticDatasetProvider, DjangoUnlabeledProvider):
    def __init__(self, project: Project, esc) -> None:
        super().__init__(project, esc)
        self.labeled = DjangoBucketProvider(
            project, NamedProvider.LABELED, ROOT_USER)

    def __contains__(self, key: int) -> bool:
        return super().__contains__(key) and not self.labeled.__contains__(key)

    def search_query(self, terms: str, exclude_list: List[int], size: int) -> List[int]:
        exclude_es_identifiers = list(self._documents.filter(
            pk__in=exclude_list).values_list("identifier", flat=True))
        labeled_identifiers = list(self.labeled._documents.values_list("identifier", flat=True).all())
        exclude_es_identifiers = exclude_es_identifiers + labeled_identifiers
        es_identifiers = self._esc.search_query(terms, exclude_es_identifiers, size)
        docs = list(self._documents.filter(identifier__in=es_identifiers).values_list("pk", flat=True).all())
        return docs

    @property
    def _documents(self) -> QuerySet:
        return super()._documents.exclude(pk__in=self.labeled.key_list)
