from __future__ import annotations

import collections
import logging
from typing import Any, Callable, Generic, Optional, Sequence, Set, TypeVar
from typing_extensions import Self

from instancelib import Instance

from ..analysis.base import AnnotationStatistics
from ..environment import AbstractEnvironment
from ..typehints import DT, IT, KT, LT, RT, VT
from .base import ActiveLearner

FT = TypeVar("FT")
F = TypeVar("F", bound=Callable[..., Any])

LOGGER = logging.getLogger(__name__)


class PoolBasedAL(
    ActiveLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    """PoolBasedAL specifies the basis for a poolbased Active Learning algorithm.
    Any algorithm that determines an ordering for *n* instances, can be based on
    this class.

    Examples
    --------
    Creation and initialization by attaching an Environment `env`.

    >>> al = PoolBasedAL(env)

    Instances can be sampled as follows

    >>> instance = next(al)

    Or in batch mode by using :func:`itertools.islice`:

    >>> instances = itertools.islice(al, 10) # Change the number to get more instances

    Mark a document as labeled

    >>> al.set_as_labeled(instance)

    Mark a document as unlabeled

    >>> al.set_as_unlabeled(instance)

    Update the ordering

    >>> al.update_ordering()

    Check how many documents are labeled

    >>> al.len_labeled
    """

    sampled: Set[KT]
    """The documents that were sampled during by this learner"""

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        self.env = env
        self.ordering = None
        self._stats = AnnotationStatistics()
        self.sampled: Set[KT] = set()
        self.identifier = identifier

    def _set_ordering(self, ordering: Sequence[KT]) -> None:
        """Set the ordering of the learner and clear the sampled set

        Parameters
        ----------
        ordering : Sequence[KT]
            The new ordering
        """
        LOGGER.info("Setting a new ordering for %s", self.name)
        self.ordering = collections.deque(ordering)  # type: ignore
        self.sampled.clear()

    def update_ordering(self) -> bool:
        ordering = self.env.unlabeled.key_list
        self._set_ordering(ordering)
        return True

    @property
    def has_ordering(self) -> bool:
        """Check if the learner has an ordering

        Returns
        -------
        bool
            True if it has an ordering
        """
        return self.ordering is not None

    @ActiveLearner.iterator_log
    def __next__(self) -> IT:
        """Query the next instance according to the ordering.
        If the instances has already been sampled during the
        current iteration. The sampled set is cleared after
        each :meth:`.update_ordering()` call

        Returns
        -------
        Instance[KT, DT, VT, RT]
            The next instance that should be labeled

        Raises
        ------
        StopIteration
            If there are no instances left that have not been sampled
        NoOrderingException
            If the ordering was not determined


        Examples
        --------
        :meth:`~allib.activelearning.PoolBasedAL.__next__()` is a special method and
        makes this object combined with the method :meth:`~allib.base.ActiveLearner.__iter__()`
        an :term:`iterator`.
        As any iterator, this object can be used as follows:

        >>> # Initialize the object
        >>> al = PoolBasedAL(env)
        >>> # Request the most informative instance
        >>> ins = next(al)
        >>> # Request the 10 most informative instances
        >>> inss = itertools.islice(al, 10)
        """
        if self.ordering is not None:
            try:
                # Pop the most left item from the deque
                key = self.ordering.popleft()
                # If it has already been sampled, repeat
                while key in self.sampled:
                    key = self.ordering.popleft()
            except IndexError:
                # There are no more items left, raise StopIteration so iterator knows it should stop
                raise StopIteration()
            else:
                # Mark the instance as sampled
                self.sampled.add(key)
                # Retrieve the instance from the dataset
                result = self.env.dataset[key]
                return result
        raise StopIteration()

    @ActiveLearner.label_log
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as labeled

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be marked as labeled
        """
        self.env.unlabeled.discard(instance)
        self.env.labeled.add(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be marked as unlabeled
        """
        self.env.labeled.discard(instance)
        self.env.unlabeled.add(instance)

    @classmethod
    def builder(cls, *args: Any, **kwargs: Any) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT], *_: Any, **__: Any
        ) -> Self:
            return cls(env)

        return builder_func
