from __future__ import annotations

import os

from typing import Generic, TypeVar, Optional, List, Callable, Any

import functools
import pickle
import uuid

MT = TypeVar("MT")

F = TypeVar('F', bound=Callable[..., Any])

class SaveableInnerModel(Generic[MT]):
    _name = "SaveableInnerModel"

    def __init__(self, innermodel: Optional[MT], storage_location: Optional[str], filename: Optional[str] = None, taboo_fields: Optional[List[str]] = None):
        self.storage_location = storage_location
        self.saved = False
        self.innermodel = innermodel
        if filename is None:
            self.filename = self._generate_random_file_name()
        else:
            self.filename = filename
        self.taboo_fields = {"innermodel": None}
        if taboo_fields is not None:
            for field in taboo_fields:
                self.taboo_fields[field] = None

    def _generate_random_file_name(self) -> str:
        """Generates a random filename

        Returns
        -------
        str
            A random file name
        """
        gen_uuid = uuid.uuid4()
        filename = f"classifier_{self._name}_{gen_uuid}.data"
        return filename

    @property
    def filepath(self) -> Optional[str]:
        if self.storage_location is not None:
            return os.path.join(self.storage_location, self.filename)
        return None

    @property
    def has_storage_location(self) -> bool:
        return self.storage_location is not None

    @property
    def is_stored(self) -> bool:
        return self.saved

    @staticmethod
    def load_model_fallback(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: SaveableInnerModel[MT], *args: Any, **kwargs: Any) -> Any:
            if not self.is_loaded and self.is_stored:
                self.load()
            return func(self, *args, **kwargs)
        return wrapper # type: ignore

    @property
    def is_loaded(self) -> bool:
        return self.innermodel is not None

    def __getstate__(self):
        if not self.has_storage_location:
            return self.__dict__
        self.save()
        state = {key: value for (key, value) in self.__dict__.items(
        ) if key not in self.taboo_fields}
        state = {**state.copy(), **self.taboo_fields}
        return state

    def save(self) -> None:
        assert self.filepath is not None
        with open(self.filepath, mode="wb") as filehandle:
            pickle.dump(self.innermodel, filehandle)
        self.saved = True

    def load(self) -> None:
        assert self.filepath is not None
        with open(self.filepath, mode="rb") as filehandle:
            self.innermodel = pickle.load(filehandle)
