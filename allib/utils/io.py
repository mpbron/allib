from os import PathLike
from pathlib import Path
from typing import Union


def create_dir_if_not_exists(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.exists():
       path.mkdir(parents=True)