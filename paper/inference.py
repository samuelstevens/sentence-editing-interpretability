"""
Given a validation dataset, writes predictions to a file so that they can be analyzed.
"""
import csv
import os
from pathlib import Path
from typing import List

import torch
from transformers import PreTrainedTokenizerFast

from . import disk, models, tokenizers, util
from .structures import HyperParameters
from .types import Result


def get_filename(config: HyperParameters, checkpoint_filepath: Path, ext: str) -> str:
    _, checkpoint_filename = os.path.split(checkpoint_filepath)
    checkpoint_hash, _ = os.path.splitext(checkpoint_filename)
    return f"{config.experiment_name}-{checkpoint_hash}-{ext}-inference.csv"


def simple_inference(
    sentences: List[str],
    model: models.SentenceClassificationModel,
    tokenizer: PreTrainedTokenizerFast,
) -> List[int]:
    """
    Given a list of sentences, just gives you a list of predictions (1 for edit, 0 for no edit). Very slow, do not use this except for one-off tasks.
    """
    device = util.get_device()
    model.eval()

    result: List[int] = []

    for sentence in sentences:
        cpu_ids, cpu_mask = tokenizers.encode_batch([sentence], tokenizer)

        input_ids = torch.stack(cpu_ids).to(device)
        attention_mask = torch.stack(cpu_mask).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        _, predictions = torch.max(logits.data, 1)  # type: ignore

        result.extend(predictions.tolist())

    return result


def perform_inference(
    config: HyperParameters, checkpoint_filepath: Path, test: bool = False
) -> Result[None]:

    disk_model = disk.load_finetuned_model_from_disk(config, checkpoint_filepath)
    if isinstance(disk_model, Exception):
        return disk_model

    model, _ = disk_model
    model.eval()

    util.print_device()

    tokenizer = tokenizers.get_tokenizer(config)

    examples_filepath = config.test_file if test else config.val_file
    filename = get_filename(config, checkpoint_filepath, "test" if test else "val")

    device = util.get_device()

    print("\n--- Inference ---")
    print(f"Writing to {filename}.")

    with open(filename, "w") as outfile:
        csvwriter = csv.writer(outfile)

        with open(examples_filepath, "r") as infile:
            csvreader = csv.reader(infile)
            print(next(csvreader))  # skip headers

            for batch in util.my_tqdm(util.chunks(csvreader, config.batch_size)):
                sentences = [i for _, i, _ in batch]
                cpu_labels = [int(i) for _, _, i in batch]
                identifiers = [i for i, _, _ in batch]

                cpu_ids, cpu_mask = tokenizers.encode_batch(sentences, tokenizer)

                integer_ids = [ids.tolist() for ids in cpu_ids]
                decoded_sentences = [
                    tokenizers.decode(ids, tokenizer) for ids in integer_ids
                ]

                input_ids = torch.stack(cpu_ids).to(device)
                attention_mask = torch.stack(cpu_mask).to(device)

                with torch.no_grad():
                    logits = model(input_ids, attention_mask)

                _, predictions = torch.max(logits.data, 1)  # type: ignore

                csvwriter.writerows(
                    zip(
                        identifiers, decoded_sentences, cpu_labels, predictions.tolist()
                    )
                )

    return None
