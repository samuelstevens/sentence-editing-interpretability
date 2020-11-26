import argparse
from typing import Collection, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import disk, table
from .structures import Score

# region colors

BLACK = "k"
GREEN = "#59d98e"
SEA = "#159d82"
BLUE = "#3498db"
PURPLE = "#9b59b6"
GREY = "#95a5a6"
RED = "#e74c3c"
ORANGE = "#f39c12"

# endregion


def get_relevant_scores(
    scores: List[Score],
) -> Tuple[Score, Score, Score, Score, Score, Score]:
    """
    returns a list of scores for
    * Random
    * BERT_default-sum
    * BERT_GLUE-sum
    * BERT-sum
    * SciBERT-sum
    * RoBERTA-sum
    """

    random = next(score for score in scores if score.model_name == "random")

    bert_default_sum = next(
        score
        for score in scores
        if score.model_name == "bert_default" and score.strategy == "sum"
    )

    bert_glue_sum = next(
        score
        for score in scores
        if score.model_name == "bert_glue" and score.strategy == "sum"
    )

    bert_sum = next(
        score
        for score in scores
        if score.model_name == "bert" and score.strategy == "sum"
    )

    scibert_sum = next(
        score
        for score in scores
        if score.model_name == "scibert" and score.strategy == "sum"
    )

    roberta_sum = next(
        score
        for score in scores
        if score.model_name == "roberta" and score.strategy == "sum"
    )

    return (random, bert_default_sum, bert_glue_sum, bert_sum, scibert_sum, roberta_sum)


def format_x_label(name: str, data: str) -> str:
    if data:
        return f"{name}\n({data})"
    else:
        return name


def plot_n_bars(
    x: Collection[str], y_points: Collection[Iterable[float]], y_labels: Collection[str]
) -> None:
    width = 0.7 / len(y_points)  # the width of the bars

    x_pos = np.arange(len(x))

    fig, ax = plt.subplots()

    colors = [SEA, ORANGE, BLUE]
    rects = []
    for i, (y, c) in enumerate(zip(y_points, colors)):
        rects.append(ax.bar(x_pos + i * width, y, width, color=c)[0])

    ax.set_ylabel("Top 3 Match Accuracy")
    ax.set_xticks(x_pos + (len(y_points) / 2 - 0.5) * width)
    ax.set_xticklabels(x)
    ax.legend(rects, y_labels)

    fig.tight_layout()


def main(interactive: bool) -> None:
    spell_scores = get_relevant_scores(disk.get_scores(disk.SPELLING_SCORES_FILE))

    # comma_scores = disk.get_scores(disk.COMMA_SCORES_FILE)
    # comma_scores = get_relevant_scores(comma_scores)

    del_scores = get_relevant_scores(disk.get_scores(disk.DELETE_SCORES_FILE))

    points = [
        (
            format_x_label(
                *table.model_name_to_model_and_dataset(spell_score.model_name)
            ),
            spell_score.top_3_match_accuracy,
            del_score.top_3_match_accuracy,
        )
        for spell_score, del_score in zip(spell_scores, del_scores)
    ]

    xlabels, y_spelling, y_delete = zip(*points)

    plot_n_bars(
        xlabels, (y_spelling, y_delete), ("spelling error", "deleted words"),
    )

    if interactive:
        plt.show()
    else:
        plt.savefig("./docs/images/interpretability/bar.pdf")
        plt.savefig("./docs/images/interpretability/bar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive", help="show the graph interactively", action="store_true",
    )
    args = parser.parse_args()
    main(args.interactive)
