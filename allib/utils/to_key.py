from typing import TypeVar, Union, Any

from instancelib.instances import Instance

KT = TypeVar("KT")

def to_key(instance_or_key: Union[KT, Instance[KT, Any, Any, Any]]) -> KT:
    """Returns the identifier of the instance if `instance_or_key` is an `Instance`
    or return the key if `instance_or_key` is a `KT`

    Parameters
    ----------
    instance_or_key : Union[KT, Instance]
        An implementation of `Instance` or an identifier typed variable

    Returns
    -------
    KT
        The identifer of the instance (or the input verbatim)
    """
    if isinstance(instance_or_key, Instance):
        key: KT = instance_or_key.identifier # type: ignore
        return key
    return instance_or_key