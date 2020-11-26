from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data, disk, evaluating, models, util
from .disk import Checkpoint
from .structures import ConfusionMatrix, HyperParameters
from .types import Result


def stop_early(history: List[Tuple[Path, float]]) -> bool:
    """
    Returns true if the last 5 epochs have all been more than the minimum validation loss
    """
    patience = 5

    if not history:
        return False

    min_val_loss = min([loss for _, loss in history])

    for _, epoch_val_loss in history[-patience:]:
        if epoch_val_loss <= min_val_loss:
            return False

    return True


def train(
    checkpoint: Checkpoint,
    config: HyperParameters,
    train_loader: DataLoader,  # type: ignore
) -> Result[Tuple[float, Checkpoint]]:
    """
    Tries to finish the current epoch while saving checkpoints at the appropriate times. It does not save the model at the end of training.
    """

    print("\n--- Training ---")

    device = util.get_device()

    epoch_loss = checkpoint.total_train_loss

    model = checkpoint.model
    model.train()

    optimizer = checkpoint.optimizer

    checkpoint_steps: List[int] = [
        int(len(train_loader) * config.checkpoint_interval * (i + 1))
        for i in range(int(1 / config.checkpoint_interval))
    ]

    for seen, batch in enumerate(util.my_tqdm(train_loader)):
        if seen < checkpoint.seen:
            continue  # catch up to where we started

        if seen in checkpoint_steps:
            checkpoint = Checkpoint(
                model, optimizer, checkpoint.epoch, seen, epoch_loss
            )
            disk.save_checkpoint(checkpoint, config)

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        model.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        loss.backward()  # type: ignore

        optimizer.step()

        epoch_loss += loss.detach().item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.3f}")

    checkpoint = Checkpoint(
        model, optimizer, checkpoint.epoch, len(train_loader), epoch_loss
    )

    return avg_loss, checkpoint


def validate(
    model: models.SentenceClassificationModel, val_loader: DataLoader  # type: ignore
) -> Tuple[float, ConfusionMatrix]:
    print("\n--- Validation ---")

    device = util.get_device()

    total_loss = 0.0
    total_confusion = ConfusionMatrix(0, 0, 0, 0)

    model.eval()

    for batch in util.my_tqdm(val_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        total_loss += loss.detach().item()
        total_confusion = total_confusion + evaluating.get_confusion_matrix(
            logits, labels
        )

    avg_loss = total_loss / len(val_loader)

    print(
        f"""Validation loss: {avg_loss:.3f}
Validation accuracy: {total_confusion.accuracy:.3f}
Validation F1: {total_confusion.f1:.3f}
Validation precision: {total_confusion.precision:.3f}
Validation recall: {total_confusion.recall:.3f}"""
    )

    return avg_loss, total_confusion


def train_and_validate(
    checkpoint: Checkpoint, config: HyperParameters, small: bool = False
) -> Result[None]:
    """
    Runs a single training and validation epoch, saving checkpoints where appropriate.
    """

    train_loader = data.get_train_dataloader(config, small=small)
    val_loader = data.get_val_dataloader(config, small=small)

    train_result = train(checkpoint, config, train_loader)
    if isinstance(train_result, Exception):
        return train_result

    train_loss, checkpoint = train_result
    disk.save_checkpoint(checkpoint, config)

    val_result = validate(checkpoint.model, val_loader)

    if isinstance(val_result, Exception):
        return val_result

    val_loss, _ = val_result
    checkpoint.avg_val_loss = val_loss
    checkpoint.epoch += 1
    checkpoint.seen = 0
    checkpoint.total_train_loss = 0
    disk.save_checkpoint(checkpoint, config)

    return None


def sanity_check(config: HyperParameters) -> Result[None]:
    util.print_device()

    checkpoint = disk.new_checkpoint(config)
    if isinstance(checkpoint, Exception):
        return checkpoint

    result = train_and_validate(checkpoint, config, small=True)
    if isinstance(result, Exception):
        return result

    next_checkpoint = disk.load_latest_checkpoint(config)
    if isinstance(next_checkpoint, Exception):
        return next_checkpoint

    assert next_checkpoint.epoch == 1
    assert next_checkpoint.seen == 0

    # continue training
    result = train_and_validate(next_checkpoint, config, small=True)
    if isinstance(result, Exception):
        return result

    return None


def main_loop(config: HyperParameters, continuing: bool) -> Result[None]:
    util.print_device()

    if continuing:
        checkpoint = disk.load_latest_checkpoint(config)

        if isinstance(checkpoint, Exception):
            return checkpoint
    else:
        checkpoint = disk.new_checkpoint(config)

    for epoch in range(config.max_epochs):
        if epoch < checkpoint.epoch:
            continue

        print(f"Training epoch {epoch}.")
        result = train_and_validate(checkpoint, config)
        if isinstance(result, Exception):
            return result

        # check if early stopping should occur.
        history = disk.load_training_history(config)
        if isinstance(history, Exception):
            return history

        if stop_early(history):
            print("Stopping early")
            break

        checkpoint = disk.load_latest_checkpoint(config)
        if isinstance(checkpoint, Exception):
            return checkpoint

    return None
