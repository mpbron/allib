# Copyright (C) 2021 The ALLib Authors. All Rights Reserved.

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

from typing import Any, TypeVar

from instancelib import Instance

KT = TypeVar("KT") # The Key Type of the Dataset (mostly int for Primary Keys, but strings are also possible)
DT = TypeVar("DT") # The Type of the data point
VT = TypeVar("VT") # The Type of the vector
RT = TypeVar("RT") # The Type of the representation of the data point
LT = TypeVar("LT") # The Type of the labels
CT = TypeVar("CT") # The Type of the context
LVT = TypeVar("LVT") # The Type of the encoded label vector
PVT = TypeVar("PVT") # The Type of the probability vector
IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)
