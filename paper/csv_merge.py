import csv
import sys
from pathlib import Path
from typing import IO, Dict, Generic, Iterator, List, NamedTuple, Optional, Tuple

from .types import T


class Peekable(Generic[T]):
    peeked: Optional[T]
    it: Iterator[T]

    def __init__(self, it: Iterator[T]):
        self.peeked = None
        self.it = it

    def peek(self) -> Optional[T]:
        if not self.peeked:
            try:
                self.peeked = next(self.it)
            except StopIteration:
                return None
        return self.peeked

    def pop(self) -> Optional[T]:
        if self.peeked:
            result = self.peeked
            self.peeked = None
            return result
        else:
            try:
                return next(self.it)
            except StopIteration:
                return None

    def __iter__(self) -> Iterator[T]:
        return self.it


class Option(NamedTuple):
    length: int
    identifier: str
    sentence: str
    label: str
    path: Path


def sorted_csv_files_reader(files: List[Path]) -> Iterator[Tuple[str, int]]:
    opened_files: Dict[Path, IO[str]] = {}
    csv_readers: Dict[Path, Peekable[List[str]]] = {}

    for filepath in files:
        opened_file = open(filepath, "r")
        opened_files[filepath] = opened_file

        reader = csv.reader(opened_file)
        next(reader)  # skip header

        csv_readers[filepath] = Peekable(reader)

    while len(opened_files) > 0:
        best = Option(sys.maxsize, "", "", "", Path())
        finished_files = set()

        for path, peekable in csv_readers.items():
            top = peekable.peek()

            if not top:
                finished_files.add(path)
                continue  # should close the file somehow

            identifier, sent, label = top

            if len(sent) < best.length:
                best = Option(len(sent), identifier, sent, label, path)

        # close any finished files:
        for path in finished_files:
            del csv_readers[path]
            opened_files[path].close()
            del opened_files[path]

        # now return the best option and pop it off the iterator
        if best.path not in csv_readers:
            return

        csv_readers[best.path].pop()

        yield best.sentence, int(best.label)
