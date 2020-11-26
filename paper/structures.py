# data_structures.py

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from typing_extensions import Literal


class HyperParameters:
    batch_size: int
    learning_rate: float
    epochs_in_12_hours: int
    train_file: Optional[Path]
    train_folder: Optional[Path]
    train_preprocessed_folder: Path
    val_file: Path
    val_preprocessed_folder: Path
    test_file: Path
    test_preprocessed_folder: Path
    model_name: str
    root_model_name: Optional[str]
    tokenizer_name: str
    models_dir: Path
    experiment_name: str
    checkpoint_interval: float
    vocab_size: int
    checkpoint_csv: Path

    def __init__(self, toml_filepath: str) -> None:
        parsed = toml.load([toml_filepath])

        self.batch_size = int(parsed["batch_size"])
        self.learning_rate = float(parsed["learning_rate"])
        self.max_epochs = int(parsed["max_epochs"])
        self.model_name = parsed["model_name"]
        self.tokenizer_name = parsed["tokenizer_name"]
        self.val_file = Path(parsed["val_file"])
        self.test_file = Path(parsed["test_file"])
        self.experiment_name = parsed["experiment_name"]
        self.models_dir = Path(parsed["models_dir"])
        self.checkpoint_interval = float(parsed["checkpoint_interval"])

        self.train_file = None
        self.train_folder = None
        if "train_file" in parsed:
            self.train_file = Path(parsed["train_file"])
        elif "train_folder" in parsed:
            self.train_folder = Path(parsed["train_folder"])
        else:
            raise ValueError(
                "Must provide either a train_file or a train_folder in .toml file."
            )

        self.root_model_name = None
        if "root_model_name" in parsed:
            self.root_model_name = parsed["root_model_name"]

        self.train_preprocessed_folder = self.get_preprocessed_folder("train")
        self.val_preprocessed_folder = self.get_preprocessed_folder("validate")
        self.test_preprocessed_folder = self.get_preprocessed_folder("test")

        # where will we store the checkpoints file?
        directory, _ = os.path.split(toml_filepath)
        checkpoint_csv_filename = f"{self.experiment_name}-checkpoints.csv"
        self.checkpoint_csv = Path(directory) / checkpoint_csv_filename

    def get_preprocessed_folder(
        self, kind: Literal["train", "validate", "test"]
    ) -> Path:
        suffix = "-preprocessed"
        if kind == "train":
            if self.train_file:
                name, ext = os.path.splitext(self.train_file)
                name += suffix
                assert not os.path.isfile(name)
                os.makedirs(name, exist_ok=True)
                return Path(name)
            elif self.train_folder:
                return self.train_folder.with_name(self.train_folder.name + suffix)
            else:
                raise ValueError(
                    "Must provide either a train_file or a train_folder in .toml file."
                )
        elif kind == "validate":
            name, ext = os.path.splitext(self.val_file)
            name += "-preprocessed"
            assert not os.path.isfile(name)
            os.makedirs(name, exist_ok=True)
            return Path(name)
        elif kind == "test":
            name, ext = os.path.splitext(self.test_file)
            name += "-preprocessed"
            assert not os.path.isfile(name)
            os.makedirs(name, exist_ok=True)
            return Path(name)
        else:
            raise ValueError(f"{kind} is not one of 'train', 'validate' or 'test'.")


@dataclass
class ConfusionMatrix:
    TP: int
    TN: int
    FP: int
    FN: int

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.TP + other.TP,
            self.TN + other.TN,
            self.FP + other.FP,
            self.FN + other.FN,
        )

    @property
    def accuracy(self) -> float:
        return (self.TP + self.TN) / self.total

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall

        if denominator == 0:
            print("Both precision and recall were 0; not going to divide by 0.")
            return 0

        return (2 * self.precision * self.recall) / (self.precision + self.recall)

    @property
    def recall(self) -> float:
        if self.TP + self.FN == 0:
            # means there are no negative examples (unlikely)
            print("No negative examples provided; not going to divide by 0.")
            return 1
        return self.TP / (self.TP + self.FN)

    @property
    def precision(self) -> float:
        if self.TP + self.FP == 0:
            # means we didn't guess positive for any examples
            return 0
        return self.TP / (self.TP + self.FP)

    @property
    def total(self) -> int:
        return self.TP + self.FP + self.TN + self.FN
