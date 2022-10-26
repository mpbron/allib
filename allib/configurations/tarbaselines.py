from typing import Any, Dict
from ..module import ModuleCatalog as Cat

autotar = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.AUTOTAR,
    "k_sample": 100,
    "batch_size": 20, 
}

autostop = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.AUTOSTOP,
    "k_sample": 100,
    "batch_size": 1, 
}