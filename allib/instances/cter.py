from __future__ import annotations

import json
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterable, Iterator, Optional, List

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from gensim.parsing.preprocessing import (preprocess_string,
                                          strip_multiple_whitespaces,
                                          strip_punctuation)
from pandas import json_normalize

from elastic_scripts.queries import es_query_featurecontext
from elastic_scripts.makecontext import getcontext
from instances.base import ContextInstance
from instances.django import DjangoBucketProvider, DjangoInstance
from instances.elastic import (ElasticDatasetProvider, ElasticLazy, ElasticConnection,
                               ElasticProvider, ElasticUnlabeledProvider)
from teacher_api.models import Document, Project, User


def es_doc_query(identifier) -> Dict[str, str]:
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
        "_source": ["identifier", "text"],
        "query": {
            "bool": {
                "must": {"match": {"identifier": identifier}}
            }
        }
    }
    return query_doc_text


def nice_date(dt, time_only=False):
    """
    :param dt: datetime string in the format 2018-05-03T10:54:40.000Z
    :param time_only: whether or not only the time should be returned (hour and minutes)
    :return: a nice string in the format 03 May 2018 - 10:54
    """
    try:
        datetime_obj = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        datetime_obj = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ')
    if not time_only:
        string = datetime_obj.strftime('%d-%m-%Y')
    else:
        string = datetime_obj.strftime('%H:%M')
    return string


def get_simple_user_string(user, user_dict):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    try:
        return user_dict[user], user_dict
    except KeyError:
        user_dict[user] = 'PERSOON-%s' % (alphabet[len(user_dict.keys())])
        return user_dict[user], user_dict


def makestring(myrow, row_user, switch, first, n_switches):
    """
    Make a nice string of a row of the context of a retrieved message from Elastic in a nice readable format
    """
    text = myrow['_source.text']
    eventdate = myrow['_source.eventDate']
    textrow = ''
    even = n_switches % 2 == 0
    if first:
        textrow = row_user + " \t " + nice_date(eventdate) + "\n\n"
        textrow += text + "\n" + nice_date(eventdate, time_only=True) + "\n"
    elif switch and not even:
        textrow = '[SWITCH]' + row_user + \
            " \t " + nice_date(eventdate) + "\n\n"
        textrow += text + "\n" + nice_date(eventdate, time_only=True) + "\n"
    elif switch and even:
        textrow += '[SWITCH]' + \
            nice_date(eventdate) + " \t " + row_user + "\n\n"
        textrow += text + "\n" + nice_date(eventdate, time_only=True) + "\n"
    else:
        textrow += '\n' + text + "\n" + \
            nice_date(eventdate, time_only=True) + "\n"

    return textrow


def nice_chat_texts(mydf):
    """
    Takes a chat context dataframe from getcontext and makes it into a nicely formatted string for the front end
    """
    text = ''
    prev_user = ''
    switches = []
    user_dict = {}
    for _, row in mydf.iterrows():
        current_user = row['_source.actor.userIdentifier']
        current_user, user_dict = get_simple_user_string(
            current_user, user_dict)
        switch = current_user != prev_user
        switches.append(switch)
        text += makestring(row, current_user, switch,
                           prev_user == '', switches.count(True))
        prev_user = current_user
    return text


class CterInstance(DjangoInstance, ContextInstance):
    def __init__(
            self,
            identifier: int,
            data: str,
            context: str,
            representation: str,
            vector) -> None:
        super().__init__(identifier, data, vector)
        self._context = context
        self._representation = representation
        self._vector = vector

    @property
    def context(self) -> pd.DataFrame:
        return self._context

    @property
    def representation(self) -> str:
        return self._representation

    @classmethod
    def from_doc_and_es(cls, doc: Document, text: str, context: str, representation: str):
        return cls(
            doc.pk,
            text,
            context,
            representation,
            doc.vector)


class ContextConnection(ElasticConnection):
    def get_doc_context(self, identifier: str) -> pd.DataFrame:
        """
        Heeft een document identifier uit Elastic nodig, en geeft een dataframe met
        indien mogelijk 11 berichten terug.
        Uitgaande van de identifier wordt geprobeerd om van de vorige en
        de volgende dag de vorige en volgende 5 berichten te vinden, samen 13 berichten.
        Het dataframe bestaat uit de kolommen, en is gesorteerd op tijd.
        """
        context_dict = getcontext([identifier], self.connection, project_str=self.project.name)[0]
        dfcontext = context_dict['context']
        return dfcontext

    def get_nice_elastic_chat_context(self, identifier: str) -> str:
        context_df = self.get_doc_context(identifier)
        context_str = nice_chat_texts(context_df)
        return context_str

    def get_message_context(self, identifier: str) -> str:
        context_df = self.get_doc_context(identifier)
        context_str = ' '.join(list(context_df.loc[:, '_source.text']))
        return context_str

    def get_feature_df(self, quick_mode=False):
        if quick_mode:
            #es.search should take a connection, not be one!???
            es_results = self.connection.search(
                body=es_query_featurecontext(),
                index=self.index, size=1000)['hits']['hits']
        else:
            es_results = scan(
                self.connection,
                query=es_query_featurecontext(),
                index=self.index,
                size=1000)
        es_data = json_normalize(es_results)
        es_data.rename(columns={'_id': 'identifier', '_source.text': 'text',
                                   '_source.feature.context': 'context'}, inplace=True)
        return es_data


class CterLazyInstance(ElasticLazy, ContextInstance):
    def __init__(
            self, 
            esc: ContextConnection,
            identifier: int,
            elastic_identifier: str, 
            data = None, context = None, 
            representation = None, vector = None) -> None:
        super().__init__(esc, identifier, elastic_identifier, data, vector)
        self._context = context
        self._representation = representation

    @property
    def context(self) -> str:
        if self._context is None:
            self._context = self._esc.get_message_context(self.elastic_identifier)
        return self._context

    @property
    def representation(self) -> str:
        if self._representation is None:
            self._representation = self._esc.get_nice_elastic_chat_context(self.elastic_identifier)
        return self._representation


class CterProvider(ElasticProvider, ABC):
    @abstractmethod
    def __init__(self, project: Project, esc: ContextConnection) -> None:
        super().__init__(project, esc)
        self._feature_matrix = None
        
    def _build_instance(self, key: int) -> ContextInstance:
        doc = Document.objects.get(pk=key)
        return self._build_instance_from_doc(doc)

    def _build_instance_from_doc(self, doc: Document) -> ContextInstance:
        return CterLazyInstance(self._esc, 
            doc.pk, doc.identifier, None, None, None, doc.vector)

    def get_data(self) -> pd.DataFrame:
        document_tuples = list(
            self._documents.all().values_list("pk", "identifier"))
        django_df = pd.DataFrame(document_tuples, columns=['pk', 'identifier'])
        feature_df = self._esc.get_feature_df()
        project_df = pd.merge(django_df, feature_df,
                              on='identifier', how='inner')
        project_df['context'] = project_df['context'].apply(
                lambda x: json.loads(x)['text'])
        return project_df[['pk', 'identifier', 'text', 'context']].drop_duplicates()

    def bulk_get_all(self) -> List[ContextInstance]:
        df = self.get_data()
        results = []
        for _, row in df.iterrows():
            results.append(
                CterInstance(int(row["pk"]), row["text"], row["context"], "", None))
        return results


class CterBucketProvider(CterProvider, DjangoBucketProvider):
    def __init__(self,
                 project: Project,
                 identifier: str,
                 user: User,
                 esc: ContextConnection) -> None:
        super().__init__(project, esc)
        self.identifier = identifier
        self.user = user
        self._feature_matrix = None


class CterDatasetProvider(CterProvider, ElasticDatasetProvider):
    def __init__(self, project: Project, esc: ContextConnection) -> None:
        super().__init__(project, esc)


class CterUnlabeledProvider(CterProvider, ElasticUnlabeledProvider):
    def __init__(self, project: Project, esc: ContextConnection) -> None:
        super().__init__(project, esc)


