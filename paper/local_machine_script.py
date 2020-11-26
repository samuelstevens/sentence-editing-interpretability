"""
Creates a command-line script for local use (typically windows).
"""

import argparse
import cProfile
import os
import shutil
from pathlib import Path

import torch

from . import data, disk, inference, models, tokenizers, training, util
from .structures import HyperParameters

assert (
    not util.is_interactive()
), "Should not be inside a iPython notebook for this script."


def show_error(err: Exception) -> None:
    print("Error:")
    print(f"\t{err}\n")


def gpu_check(force_cpu: bool) -> None:
    if not force_cpu:
        assert (
            torch.cuda.is_available()
        ), "Need CUDA to run on a local machine: https://pytorch.org/get-started/locally/"


def script() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess", help="preprocess data (tokenizer, pickle)", type=str
    )
    parser.add_argument(
        "--sanity-check", help="run the sanity check", action="store_true"
    )
    parser.add_argument(
        "--main-loop", help="run the main training loop", action="store_true"
    )
    parser.add_argument(
        "--continuing", help="is training continuing?", action="store_true"
    )
    parser.add_argument(
        "--force-cpu",
        help="force the script to use CPU is no GPU is present.",
        action="store_true",
    )
    parser.add_argument(
        "--check-loader",
        help="iterate through the train and validation dataloaders to make sure they work.",
        action="store_true",
    )
    parser.add_argument(
        "--profile",
        help="use cProfile to profile program. Saves file to PROFILE.prof",
        type=str,
    )
    parser.add_argument(
        "--inference-val",
        help="write predictions for the validation set using a checkpoint .pt file.",
        type=str,
    )
    parser.add_argument(
        "--inference-test",
        help="write predictions for the test set using a checkpoint .pt file.",
        type=str,
    )
    parser.add_argument(
        "--pretrain-prep",
        help="Do project-specific steps for additional pretraining",
        action="store_true",
    )
    parser.add_argument(
        "--copy-models", help="Copy required models to COPY_MODELS", type=str,
    )

    parser.add_argument(
        "--eval", help="evaluate checkpoint EVAL on the validation set", type=str,
    )

    parser.add_argument("config", help="location of experiment .toml config file")

    args = parser.parse_args()

    config = HyperParameters(args.config)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    if args.copy_models:
        dst_dir = Path(args.copy_models) / config.models_dir
        os.makedirs(dst_dir, exist_ok=True)

        if "arxiv" in config.model_name:
            # need to copy the custom model folder AND the latest checkpoint
            model_dir = config.models_dir / config.model_name
            assert os.path.isdir(model_dir), f"{model_dir} needs to exist."

            print(f"Copying {model_dir} to {dst_dir / config.model_name}.")
            shutil.copytree(model_dir, dst_dir / config.model_name)

        # need to copy the latest checkpoint to the models folder.
        latest_model = disk.find_latest_checkpoint(config.checkpoint_csv)
        if isinstance(latest_model, Exception):
            show_error(latest_model)
        else:
            shutil.copy(latest_model.filepath, dst_dir)

    if args.preprocess:
        types = args.preprocess.split(",")
        tokenizer = tokenizers.get_tokenizer(config)

        for t in types:
            if t not in ["train", "val", "test"]:
                print("--preprocess only accepts 'train', 'val' or 'test'.")

        if "test" in types:
            print("Preprocessing test data...")
            data.prepare_test_data(config, tokenizer)
        if "val" in types:
            print("Preprocessing validation data...")
            data.prepare_val_data(config, tokenizer)
        if "train" in types:
            print("Preprocessing training data...")
            data.prepare_train_data(config, tokenizer)

    if args.check_loader:
        print("Checking small validation loader...")
        loader = data.get_val_dataloader(config, small=True)
        for batch in util.my_tqdm(loader):
            pass
        print("Checking small training loader...")
        loader = data.get_train_dataloader(config, small=True)
        for batch in util.my_tqdm(loader):
            pass
        print("Checking validation loader...")
        loader = data.get_val_dataloader(config)
        for batch in util.my_tqdm(loader):
            pass
        print("Checking training loader...")
        loader = data.get_train_dataloader(config)
        for batch in util.my_tqdm(loader):
            pass

    if args.pretrain_prep:
        models.pretrain_prep(config)

    if args.sanity_check:
        gpu_check(args.force_cpu)
        result = training.sanity_check(config)
        if isinstance(result, Exception):
            show_error(result)

    if args.main_loop:
        gpu_check(args.force_cpu)
        result = training.main_loop(config, bool(args.continuing))
        if isinstance(result, Exception):
            show_error(result)

    if args.inference_val:
        gpu_check(args.force_cpu)
        result = inference.perform_inference(config, args.inference_val, test=False)
        if isinstance(result, Exception):
            show_error(result)

    if args.inference_test:
        gpu_check(args.force_cpu)
        result = inference.perform_inference(config, args.inference_test, test=True)
        if isinstance(result, Exception):
            show_error(result)

    if args.eval:
        val_loader = data.get_val_dataloader(config)
        checkpoint = disk.load_checkpoint(config, Path(args.eval))
        if isinstance(checkpoint, Exception):
            show_error(checkpoint)
        else:
            training.validate(checkpoint.model, val_loader)

    if args.profile:
        pr.disable()
        pr.dump_stats(args.profile + ".prof")
