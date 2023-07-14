# Copyright (C) 2021 The InstanceLib Authors. All Rights Reserved.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

from typing import Any, Callable, FrozenSet, Generic, Mapping, Optional, Sequence, Iterable, Union
from typing_extensions import Self


from ..utils.func import list_unzip3

from ..instances.abstracts import MemoryPaperAbstractInstance, MemoryPaperAbstractInstanceProvider
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.environment.memory import MemoryEnvironment
from instancelib.ingest.spreadsheet import instance_extractor, text_concatenation, no_vector, id_index
from uuid import UUID

from ..typehints import KT, VT, LT

import pandas as pd

def text_builder(
    identifier: KT,
    data: Mapping[str, str],
    vector: VT,
    representation: str,
    row: pd.Series,
    idx: Any,
) -> MemoryPaperAbstractInstance[KT, VT]:
    return MemoryPaperAbstractInstance(identifier, data, vector, representation)

def text_dict(title_col: str, abstract_col: str) -> Callable[[pd.Series], Mapping[str, str]]:
    def callable(row: pd.Series) -> Mapping[str, str]:
        mapping = {"title": row[title_col], "abstract": row[abstract_col]}
        return mapping
    return callable

def transform_labels(label_col: str, pos_label: LT, neg_label: LT) -> Callable[[pd.Series], FrozenSet[LT]]:
    def func(row: pd.Series) -> FrozenSet[LT]:
        if row[label_col] == 1:
            return frozenset([pos_label])
        return frozenset([neg_label])
    return func

class PaperAbstractEnvironment(
    MemoryEnvironment[
        MemoryPaperAbstractInstance[KT, VT], Union[KT, UUID], Mapping[str, str], VT, str, LT
    ],
    Generic[KT, VT, LT],
):
    @classmethod
    def from_data(
        cls,
        target_labels: Iterable[LT],
        indices: Sequence[KT],
        data: Sequence[Mapping[str, str]],
        ground_truth: Sequence[Iterable[LT]],
        vectors: Optional[Sequence[VT]],
    ) -> Self:
        dataset = MemoryPaperAbstractInstanceProvider[KT, VT].from_data_and_indices(
            indices, data, vectors
        )
        truth = MemoryLabelProvider[Union[KT, UUID], LT].from_data(
            target_labels, indices, ground_truth
        )
        return cls(dataset, truth)
    

        
    
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        title_col: str,
        abstract_col: str,
        label_col: str,
        pos_label: LT,
        neg_label: LT
    ):
        triples = instance_extractor(
                    df,
                    id_index(),
                    text_dict(title_col, abstract_col),
                    no_vector(),
                    text_concatenation(title_col, abstract_col),
                    transform_labels(label_col, pos_label, neg_label),
                    text_builder,
                )
        keys, instances, labels = list_unzip3(triples)
        dataset = MemoryPaperAbstractInstanceProvider(instances)
        labels = MemoryLabelProvider.from_tuples(list(zip(keys, labels)))
        return cls(dataset, labels)