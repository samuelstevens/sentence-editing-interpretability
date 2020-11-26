from typing import TypeVar, Union

T = TypeVar("T")

Result = Union[T, Exception]
