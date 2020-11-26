# util.py

import itertools
from typing import Any, Iterable, Iterator, List, Optional, Set

import torch

from .types import T


def grouper(
    iterable: Iterable[T], n: int, fillvalue: Optional[T] = None
) -> Iterator[List[T]]:
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def chunks(elements: Iterator[T], n: int) -> Iterator[List[T]]:
    """
    Yield successive n-sized chunks from elements.
    """

    while True:
        chunk = []
        for i in range(n):
            try:
                chunk.append(next(elements))
            except StopIteration:
                yield chunk
                return

        yield chunk


def get_device() -> torch.device:  # type: ignore
    if torch.cuda.is_available():
        return torch.device("cuda")  # type: ignore
    else:
        return torch.device("cpu")  # type: ignore


def print_device() -> None:
    if torch.cuda.is_available():
        print("We will use:", torch.cuda.get_device_name(0))
    else:
        print("We will use the CPU.")


def is_interactive() -> bool:
    import __main__ as main  # type: ignore

    return not hasattr(main, "__file__")


def is_py38() -> bool:
    import sys

    major, minor, micro, _, _ = sys.version_info

    return major == 3 and minor >= 8


def import_tqdm() -> Any:
    if is_interactive():
        return __import__("tqdm.notebook").tqdm
    else:
        return __import__("tqdm").tqdm


my_tqdm = import_tqdm()


def fix_special_text(text: str) -> str:

    return (
        text.replace("[MATH]", "_MATH_")
        .replace("[REF]", "_REF_")
        .replace("[EQUATION]", "_MATHDISP_")
        .replace("[CITATION]", "_CITE_")
    )


def get(s: Set[T]) -> T:
    return next(iter(s))
