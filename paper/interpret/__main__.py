import argparse
import sys
from typing import Collection, cast

from typing_extensions import Literal

from . import run

valid_types = ["comma", "delete", "spelling"]
ClassType = Literal["comma", "delete", "spelling"]


def clean_types(types: Collection[str]) -> Collection[ClassType]:
    for t in types:
        if t not in valid_types:
            print(f"type must be one of {valid_types}, not {t}.")
            sys.exit(1)

    return cast(Collection[ClassType], types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-types",
        help="comma, delete, or spelling, comma-separated, to evaluate the attention on",
        type=str,
    )
    parser.add_argument(
        "--weight-types",
        help="comma, delete, or spelling, comma-separated, to save the weights for",
        type=str,
    )
    args = parser.parse_args()

    weight_types: Collection[ClassType] = []
    eval_types: Collection[ClassType] = []

    if args.weight_types:
        weight_types = clean_types(args.weight_types.split(","))
    if args.eval_types:
        eval_types = clean_types(args.eval_types.split(","))

    if len(weight_types) == len(eval_types) == 0:
        print("No work being done. Try --help.")
        sys.exit(1)

    for t in weight_types:
        if t == "comma":
            run.save_comma_edit_weights()

        if t == "spelling":
            run.save_spelling_edit_weights()

        if t == "delete":
            run.save_delete_edit_weights()

    for t in eval_types:
        if t == "comma":
            run.evaluate_all_strategies_on_comma_edits()

        if t == "spelling":
            run.evaluate_all_strategies_on_spelling_edits()

        if t == "delete":
            run.evaluate_all_strategies_on_delete_edits()
