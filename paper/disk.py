# disk.py

import csv
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import TypedDict

from . import models, util
from .structures import HyperParameters
from .types import Result


@dataclass
class Checkpoint:
    model: models.SentenceClassificationModel
    optimizer: torch.optim.Optimizer
    epoch: int
    seen: int
    total_train_loss: float
    filepath: Optional[str] = None
    avg_val_loss: Optional[float] = None


class DiskModel(TypedDict):
    model_state_dict: Dict[Any, Any]
    optimizer_state_dict: Dict[Any, Any]


@dataclass
class CsvRow:
    filepath: Path
    epoch: int
    seen: int
    total_train_loss: float
    avg_val_loss: Optional[float]

    def __str__(self) -> str:
        row = f"{self.filepath},{self.epoch},{self.seen},{self.total_train_loss}"

        if self.avg_val_loss is not None:
            row += f",{self.avg_val_loss}"

        return row

    @staticmethod
    def parse_raw(s: str) -> "Result[CsvRow]":
        return CsvRow.parse(s.split(","))

    @staticmethod
    def parse(row: List[str]) -> "Result[CsvRow]":
        if len(row) == 5:
            filepath, epoch, seen, train_loss, val_loss = row
            return CsvRow(
                Path(filepath),
                int(epoch),
                int(seen),
                float(train_loss),
                float(val_loss),
            )
        if len(row) == 4:
            filepath, epoch, seen, train_loss = row
            return CsvRow(
                Path(filepath), int(epoch), int(seen), float(train_loss), None
            )

        else:
            return ValueError(f"row {row} does not contain 4 or 5 elements")


def read_checkpoint_file(checkpoint_csv_filepath: Path) -> Result[List[CsvRow]]:
    """
    Reads a checkpoint file and returns the data stored in the file.
    """
    if not os.path.isfile(checkpoint_csv_filepath):
        return FileNotFoundError("no checkpoint file")

    result: List[CsvRow] = []

    with open(checkpoint_csv_filepath, "r") as file:
        reader = csv.reader(file)
        try:
            for row in reader:
                parsed = CsvRow.parse(row)
                if isinstance(parsed, Exception):
                    return parsed
                result.append(parsed)
        except Exception as err:
            return err

    return result


def find_latest_checkpoint(checkpoint_csv_filepath: Path) -> Result[CsvRow]:
    """
    returns a tuple of checkpoint filepath, epoch, seen, train_loss, val_loss
    """
    rows = read_checkpoint_file(checkpoint_csv_filepath)

    if isinstance(rows, Exception):
        return ValueError(f"couldn't find latest checkpoint -> {rows}")

    if not rows:
        return ValueError("no rows in checkpoint file")

    sorted_by_seen = sorted(rows, key=lambda row: row.seen, reverse=True)
    sorted_by_epoch = sorted(sorted_by_seen, key=lambda row: row.epoch, reverse=True)

    return sorted_by_epoch[0]


def lookup_csvrow_by_checkpoint_path(
    config: HyperParameters, checkpoint_filepath: Path
) -> Result[CsvRow]:

    rows = read_checkpoint_file(config.checkpoint_csv)

    if isinstance(rows, Exception):
        return rows

    if not rows:
        return ValueError("no rows in checkpoint file")

    lookup: Dict[Path, CsvRow] = {}

    for row in rows:
        lookup[row.filepath.resolve()] = row

    if checkpoint_filepath.resolve() not in lookup:
        return ValueError(
            f"checkpoint '{checkpoint_filepath.resolve()}' not found in {config.checkpoint_csv}"
        )
    else:
        return lookup[checkpoint_filepath.resolve()]


def new_checkpoint(config: HyperParameters) -> Checkpoint:
    epoch = 0
    seen = 0
    loss = 0.0

    model = models.get_pretrained_model(config)

    try:
        model.cuda()
    except:
        pass
    optimizer = model.get_optimizer(config)

    return Checkpoint(model, optimizer, epoch, seen, loss)


def load_finetuned_model_from_disk(
    config: HyperParameters, filepath: Path
) -> Result[Tuple[models.SentenceClassificationModel, torch.optim.Optimizer]]:
    disk_model: DiskModel

    try:
        if torch.cuda.is_available():
            disk_model = torch.load(filepath)  # type: ignore
        else:
            disk_model = torch.load(filepath, map_location=torch.device("cpu"))  # type: ignore
    except Exception as err:
        return err

    model = models.get_pretrained_model(config)
    optimizer = model.get_optimizer(config)

    model.load_state_dict(disk_model["model_state_dict"])
    model.to(util.get_device())

    optimizer.load_state_dict(disk_model["optimizer_state_dict"])

    return model, optimizer


def load_checkpoint(config: HyperParameters, filepath: Path) -> Result[Checkpoint]:
    disk_model = load_finetuned_model_from_disk(config, filepath)
    if isinstance(disk_model, Exception):
        return disk_model

    model, optimizer = disk_model

    checkpoint_info = lookup_csvrow_by_checkpoint_path(config, filepath)

    if isinstance(checkpoint_info, Exception):
        return checkpoint_info

    return Checkpoint(
        model,
        optimizer,
        checkpoint_info.epoch,
        checkpoint_info.seen,
        checkpoint_info.total_train_loss,
        avg_val_loss=checkpoint_info.avg_val_loss,
    )


def load_training_history(config: HyperParameters) -> Result[List[Tuple[Path, float]]]:
    """
    Returns tuples of a checkpoint and validation loss
    """
    rows = read_checkpoint_file(config.checkpoint_csv)

    if isinstance(rows, Exception):
        return rows
    else:
        return [(row.filepath, row.avg_val_loss) for row in rows if row.avg_val_loss]


def load_latest_checkpoint(config: HyperParameters) -> Result[Checkpoint]:
    """
    Loads the model using:
    * hyperparams (using params.toml)
    * checkpoint folder

    Returns the model and the optimizer and the current epoch, # of seen examples and the current loss
    """

    # read checkpoints.csv to see if there are any existing checkpoints
    result = find_latest_checkpoint(config.checkpoint_csv)

    if isinstance(result, Exception):
        return ValueError(f"no latest checkpoint to load -> {result}")
    else:
        if os.path.isfile(result.filepath):
            # load from filepath
            return load_checkpoint(config, result.filepath)
        else:
            return new_checkpoint(config)


def save_checkpoint(checkpoint: Checkpoint, config: HyperParameters) -> None:
    """
    Saves a model (and related garbage) to disk and updates checkpoints.csv with the new model.
    """

    assert not checkpoint.filepath, "checkpoint already has a path."

    os.makedirs(config.models_dir, exist_ok=True)

    filepath = Path(config.models_dir) / f"{uuid.uuid4()}.pt"

    disk_model: DiskModel = {
        "model_state_dict": checkpoint.model.state_dict(),
        "optimizer_state_dict": checkpoint.optimizer.state_dict(),
    }

    torch.save(disk_model, filepath)  # type: ignore

    with open(config.checkpoint_csv, "a") as file:
        csv_row = CsvRow(
            filepath,
            checkpoint.epoch,
            checkpoint.seen,
            checkpoint.total_train_loss,
            checkpoint.avg_val_loss,
        )

        file.write(f"{csv_row}\n")
