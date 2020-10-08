#%%
from typing import List, Tuple, Optional
import itertools

import numpy as np # type: ignore
import pandas as pd # type: ignore

from allib.instances import DataPoint, Instance
from allib.module.factory import MainFactory, CONFIG
from allib.environment import MemoryEnvironment
from allib import Component
from allib.activelearning import ActiveLearner
from allib.feature_extraction import BaseVectorizer
from allib.activelearning.mostcertain import LabelMaximizer

from allib.module.catalog import ModuleCatalog as Cat
# %%
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import requests
import os
import json


def auth():
    return os.environ.get("BEARER_TOKEN",  "AAAAAAAAAAAAAAAAAAAAAO5XIQEAAAAAJgsOWPwdR6UXqEB0neL9yjvG%2BQ8%3DvKZbFfXR92tFHFMa9Phb8EMEtd72o8rKCy1qdK2HmFO3tWWuIA")


def create_url():
    query = "corona lang=nl"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    tweet_fields = "tweet.fields=author_id"
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}".format(
        query, tweet_fields
    )
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()



def main():
    bearer_token = auth() 
    url = create_url()
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    print(json_response)
    twitter_data = json_response["data"]
    datapoints = [DataPoint(entry["id"], entry["text"], None) for entry in twitter_data]
    for dat in datapoints:
        print(dat.data)


if __name__ == "__main__":
    main()

# %%
