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
from abc import ABC, abstractmethod

import numpy.typing as npt
from typing import Any, Generic, Mapping, Optional, Sequence, Union
from uuid import UUID, uuid4

from ..typehints import KT, VT
from instancelib.instances.memory import AbstractMemoryProvider, DataPoint
from instancelib.instances.base import Instance


class PaperAbstractInstance(Instance[KT, Mapping[str, str], VT, str], ABC, Generic[KT, VT]):
    
    @property
    def abstract(self) -> str:
        return self.data["abstract"]
    
    @property
    def title(self) -> str:
        return self.data["title"]
    
    @property
    def representation(self) -> str:
        return f"{self.title}: {self.abstract}"
    
    


class MemoryPaperAbstractInstance(
    DataPoint[Union[KT, UUID], Mapping[str,str], VT, str],
    PaperAbstractInstance[Union[KT, UUID], VT],
    Generic[KT, VT],
):
    @property
    def representation(self) -> str:
        return f"{self.title}: {self.abstract}"


class MemoryPaperAbstractInstanceProvider(
    AbstractMemoryProvider[
        MemoryPaperAbstractInstance[KT, VT], Union[KT, UUID], Mapping[str, str], VT, str
    ],
    Generic[KT, VT],
):
    def create(self, *args: Any, **kwargs: Any):
        new_key = uuid4()
        new_instance = MemoryPaperAbstractInstance[KT, VT](new_key, *args, **kwargs)
        self.add(new_instance)
        return new_instance

    @staticmethod
    def construct(*args: Any, **kwargs: Any) -> MemoryPaperAbstractInstance[KT, VT]:
        return MemoryPaperAbstractInstance[KT, VT](*args, **kwargs)
