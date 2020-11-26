import bisect
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, Subset, TensorDataset
from transformers import PreTrainedTokenizerFast

from . import csv_merge, tokenizers, util
from .structures import HyperParameters

Row = Tuple[Tensor, Tensor, Tensor]


class RandomBatchSampler(Sampler):  # type: ignore
    """
    Batches the data into size `batch_size`, then randomly shuffles those batches. Last batch is smaller, because there might not be a full batch.
    """

    def __init__(self, data: TensorDataset, batch_size: int):
        indices = list(range(len(data)))
        batches = [list(g) for g in util.grouper(indices, batch_size)]

        *front, last = batches

        last = [i for i in last if i]  # filter None out
        random.shuffle(front)

        self.batches = front + [last]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class FileBackedSentenceDataset(TensorDataset):
    dirname: Path
    size: int
    files: List[str]
    _file_lens: List[int]
    _index_starts: List[int]
    _cached: Optional[Tuple[str, TensorDataset]]

    def __init__(
        self,
        dirname: Path,
        size: int,
        data_files: Optional[List[Path]] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        """
        Arguments:
        * dirname {str} -- the folder where the pickled TensorDatasets will be stored.
        * data_files {List[str]} -- the original csv file of (sentence, label) pairs. If provided, the sentences should be ordered by length from shortest to longest to make use of smart batches per https://mccormickml.com/2020/07/29/smart-batching-tutorial. If None or empty, then the dataset assumes that the pickled TensorDatasets already exist.
        * tokenizer {Optional[PretrainedTokenizerFast]} -- tokenizer to convert the examples in example_file
        """

        self.dirname = dirname
        self.size = size
        self.files = []
        self._file_lens = []
        self._index_starts = []
        self._cached = None

        if data_files:
            assert tokenizer is not None, "Need a tokenizer with an examples file"
            self._save_all_datasets(data_files, tokenizer)
        else:
            self.files = [
                os.path.join(dirname, f)
                for f in sorted(os.listdir(dirname), key=lambda f: int(f[:-5]))
            ]

            self._file_lens = [self.size for f in self.files]
            # the last file length might be short.
            self._file_lens[-1] = self._get_file_len(self.files[-1])

        start = 0
        for file_len in self._file_lens:
            self._index_starts.append(start)
            start += file_len

    def _save_all_datasets(
        self, data_files: List[Path], tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """
        Each datafile must be sorted by length.
        """
        # clear any existing work
        shutil.rmtree(self.dirname, ignore_errors=True)
        os.makedirs(self.dirname, exist_ok=True)

        sentences = []
        labels = []
        dataset_count = 0

        for sentence, label in util.my_tqdm(
            csv_merge.sorted_csv_files_reader(data_files)
        ):
            sentences.append(sentence)
            labels.append(label)

            if len(sentences) == self.size:
                self._save_dataset(
                    self._make_dataset(sentences, labels, tokenizer),
                    f"{dataset_count}.pckl",
                )

                dataset_count += 1
                sentences = []
                labels = []

        if sentences:
            self._save_dataset(
                self._make_dataset(sentences, labels, tokenizer),
                f"{dataset_count}.pckl",
            )

    def _make_dataset(
        self,
        sentences: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerFast,
    ) -> TensorDataset:
        """
            Arguments:
            * sentences {List[str]} -- The list of sentences that need to be tokenized and padded. This will always be exactly one batch.
            """
        assert len(sentences) == len(
            labels
        ), "Need the same number of sentences and labels"

        assert (
            len(sentences) <= self.size
        ), f"{len(sentences)} must be less than {self.size}"

        input_ids, attention_masks = tokenizers.encode_batch(sentences, tokenizer)

        input_ids_t = torch.stack(input_ids)
        attention_masks_t = torch.stack(attention_masks)
        labels_t = torch.tensor(labels)  # type: ignore

        dataset = TensorDataset(input_ids_t, attention_masks_t, labels_t)

        return dataset

    def _save_dataset(self, dataset: TensorDataset, name: str) -> None:
        filepath = os.path.join(self.dirname, name)

        self.files.append(filepath)
        self._file_lens.append(len(dataset))

        with open(filepath, "wb") as file:
            pickle.dump(dataset, file)

    def _find_file(self, index: int) -> int:
        """
        Finds the i such that self._index_start[i] <= index < self._index_start[i+1]. Since the array is sorted, we use binary search to find it.
        """
        return bisect.bisect(self._index_starts, index) - 1

    def __getitem__(self, index: int) -> Row:
        file_index = self._find_file(index)
        filename = self.files[file_index]
        dataset: TensorDataset

        if self._cached is None or filename != self._cached[0]:
            # this is called once per batch, so it's working as expected. It's just slow.
            self._set_cached_file(filename)

        name, dataset = self._cached  # type: ignore
        # (OO code sucks, I know that self._cached is no longer None because we call self._set_cached_file)

        return cast(
            Tuple[Tensor, Tensor, Tensor],
            dataset[index - self._index_starts[file_index]],
        )

    def _set_cached_file(self, filename: str) -> None:
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
            self._cached = (filename, dataset)

    def __len__(self) -> int:
        return sum(self._file_lens)

    def _get_file_len(self, filepath: str) -> int:
        with open(filepath, "rb") as f:
            contents: TensorDataset = pickle.load(f)

        return len(contents)


def prepare_val_data(
    config: HyperParameters, tokenizer: PreTrainedTokenizerFast
) -> None:
    FileBackedSentenceDataset(
        config.val_preprocessed_folder, config.batch_size, [config.val_file], tokenizer,
    )


def prepare_test_data(
    config: HyperParameters, tokenizer: PreTrainedTokenizerFast
) -> None:
    FileBackedSentenceDataset(
        config.test_preprocessed_folder,
        config.batch_size,
        [config.test_file],
        tokenizer,
    )


def prepare_train_data(
    config: HyperParameters, tokenizer: PreTrainedTokenizerFast
) -> None:
    if config.train_file:
        FileBackedSentenceDataset(
            config.train_preprocessed_folder,
            config.batch_size,
            [config.train_file],
            tokenizer,
        )
    elif config.train_folder:
        train_files = [config.train_folder / f for f in os.listdir(config.train_folder)]
        FileBackedSentenceDataset(
            config.train_preprocessed_folder, config.batch_size, train_files, tokenizer,
        )


def get_train_dataloader(
    config: HyperParameters, small: bool = False
) -> DataLoader:  # type: ignore
    train_dataset = FileBackedSentenceDataset(
        config.train_preprocessed_folder, size=config.batch_size
    )

    if small:
        train_dataset = Subset(  # type: ignore
            train_dataset, list(range(min(len(train_dataset), config.batch_size * 8)))
        )

    return DataLoader(
        train_dataset,
        pin_memory=True,
        batch_sampler=RandomBatchSampler(train_dataset, config.batch_size),
    )


def get_val_dataloader(
    config: HyperParameters, small: bool = False
) -> DataLoader:  # type: ignore
    val_dataset = FileBackedSentenceDataset(
        config.val_preprocessed_folder, size=config.batch_size
    )

    if small:
        val_dataset = Subset(val_dataset, list(range(config.batch_size * 8)))  # type: ignore

    return DataLoader(
        val_dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size
    )


def get_test_dataloader(
    config: HyperParameters, small: bool = False
) -> DataLoader:  # type: ignore
    test_dataset = FileBackedSentenceDataset(
        config.test_preprocessed_folder, size=config.batch_size
    )

    if small:
        test_dataset = Subset(test_dataset, list(range(config.batch_size * 8)))  # type: ignore

    return DataLoader(
        test_dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size
    )
