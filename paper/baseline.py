"""
Predicts the most common class for the validation set on the validation and test set (1) and reports rulests.
"""

import csv
from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from . import data, tokenizers, util
from .structures import HyperParameters


def baseline(
    config: HyperParameters,
    loader: "DataLoader[Tuple[Tensor, Tensor, Tensor]]",
    filename: str,
) -> None:
    util.print_device()
    tokenizer = tokenizers.get_tokenizer(config)

    device = util.get_device()

    print("\n--- Baseline ---")

    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)

        for batch in util.my_tqdm(loader):
            original_ids = [ids.tolist() for ids in batch[0]]
            sentences = [tokenizers.decode(ids, tokenizer) for ids in original_ids]

            labels = batch[2].to(device)

            csvwriter.writerows(zip(sentences, labels.tolist(), [1 for _ in labels],))


if __name__ == "__main__":
    config = HyperParameters("./experiments/scibert_32_1e6/params.toml")
    loader = data.get_val_dataloader(config)
    baseline(config, loader, "./data-unversioned/scibert-baseline-val-inference.csv")
