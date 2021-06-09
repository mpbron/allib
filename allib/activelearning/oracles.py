from .base import ActiveLearner
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import numpy as np # type: ignore

from instancelib.instances import Instance

from ..typehints import KT, DT, LT, RT, VT, IT

OracleFunction = Callable[[Instance, Iterable[LT]], LT]

def console_text_oracle(
        doc: Instance,
        labels: Iterable[LT]) -> List[LT]:
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

def label_loop(al: ActiveLearner[IT, KT, DT, VT, RT, LT]) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
    qstr = "Press ENTER to continue or one of the following options"
    qstr += "q => quit\n"
    qstr += "u => update ordering"
    continue_choice = input(qstr)
    if  continue_choice == "q":
        return al
    elif continue_choice == "u":
        print("Retraining the classifier...")
        al.update_ordering()
    print("Retrieving the next document... \n")
    
    labelset = al.env.labels.labelset
    label_dict = dict(enumerate(labelset, start=1))
    
    doc = next(al)
    
    lstr =  "Please label the following instance: \n"
    lstr += "==================================== \n"
    lstr += f"{doc.representation} \n"
    lstr += "==================================== \n"
    lstr += f"Document ID: {doc.identifier} \n"
    lstr += f"Vector: {doc.vector} \n"
    lstr += "==================================== \n"
    for i, label in label_dict.items():
        lstr += f"{i} => {label}\n"
    # Request labels from user 
    chosen_labels = [label_dict[int(input(lstr))]]
    
    # Update state
    al.env.labels.set_labels(doc, *chosen_labels)
    al.set_as_labeled(doc)
    return label_loop(al)
    
        

