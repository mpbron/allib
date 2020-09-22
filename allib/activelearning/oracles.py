from typing import Any, Callable, Iterable, List, Optional, TypeVar

import numpy as np

from instances import Instance

LT = TypeVar("LT")

OracleFunction = Callable[[Instance, Iterable[LT]], LT]

def console_text_oracle(
        doc: Instance,
        labels: Iterable[LT]) -> LT:
    label_dict = dict(enumerate(labels, start=1))
    qstr =  "Please label the following instance: \n"
    qstr += "==================================== \n"
    qstr += "{} \n".format(doc.representation) 
    qstr += "==================================== \n"
    qstr += "Document ID: {} \n".format(doc.identifier)
    qstr += "Vector: {} \n".format(doc.vector)
    qstr += "==================================== \n"
    for i, label in label_dict.items():
        qstr += "{} => {}\n".format(i, label)
    chosen_label = int(input(qstr))
    return [label_dict[chosen_label]]
