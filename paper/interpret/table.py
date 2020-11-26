import argparse
from pathlib import Path
from typing import List, Tuple

from . import disk
from .structures import Score

TABLE_TEX_FILE = Path("./docs/thesis/interpretability-table-contents.tex")
BASELINE_TABLE_TEX_FILE = Path(
    "./docs/thesis/baseline-interpretability-table-contents.tex"
)


def model_name_to_model_and_dataset(model_name: str) -> Tuple[str, str]:
    model_name = model_name.lower()
    if model_name == "bert_default":
        return "BERT", "n/a"
    elif model_name == "bert_glue":
        return "BERT", "GLUE"
    elif model_name == "bert":
        return "BERT", "AESW"
    elif model_name == "roberta":
        return "RoBERTA", "AESW"
    elif model_name == "scibert":
        return "SciBERT", "AESW"
    elif model_name == "random":
        return "Random", ""
    else:
        raise ValueError(f"{model_name} is not a valid name")


def scores_to_table_row(
    spell_score_mean: Score,
    spell_score_max: Score,
    del_score_mean: Score,
    del_score_max: Score,
) -> str:
    assert (
        del_score_mean.model_name
        == del_score_max.model_name
        == spell_score_mean.model_name
        == spell_score_max.model_name
    )
    assert del_score_mean.strategy == spell_score_mean.strategy == "sum"
    assert del_score_max.strategy == spell_score_max.strategy == "max"

    OPEN = "{"
    CLOSE = "}"

    model, dataset = model_name_to_model_and_dataset(del_score_mean.model_name)

    strategies = ("n/a", "n/a") if model.lower() == "random" else ("mean", "max")

    template_str = f"""\\multirow{OPEN}2{CLOSE}{OPEN}*{CLOSE}{OPEN}\\makecell[l]{OPEN}\\textbf{OPEN}{model}{CLOSE}\\\\{dataset}{CLOSE}{CLOSE} & {strategies[0]} & {spell_score_mean.exact_match_accuracy:.3f} & {spell_score_mean.jaccard_match_accuracy:.3f} & {spell_score_mean.avg_jaccard_similarity:.3f} & {spell_score_mean.top_3_match_accuracy:.3f} & {del_score_mean.exact_match_accuracy:.3f} & {del_score_mean.jaccard_match_accuracy:.3f} & {del_score_mean.avg_jaccard_similarity:.3f} & {del_score_mean.top_3_match_accuracy:.3f} \\\\
& {strategies[1]} & {spell_score_max.exact_match_accuracy:.3f} & {spell_score_max.jaccard_match_accuracy:.3f} & {spell_score_max.avg_jaccard_similarity:.3f} & {spell_score_max.top_3_match_accuracy:.3f} & {del_score_max.exact_match_accuracy:.3f} & {del_score_max.jaccard_match_accuracy:.3f} & {del_score_max.avg_jaccard_similarity:.3f} & {del_score_max.top_3_match_accuracy:.3f} \\\\ \\hline\n"""

    return template_str


def get_relevant_scores(
    model_name: str, spelling_scores: List[Score], delete_scores: List[Score]
) -> Tuple[Score, Score, Score, Score]:
    spelling_mean = next(
        score
        for score in spelling_scores
        if score.model_name == model_name and score.strategy == "sum"
    )

    spelling_max = next(
        score
        for score in spelling_scores
        if score.model_name == model_name and score.strategy == "max"
    )

    delete_mean = next(
        score
        for score in delete_scores
        if score.model_name == model_name and score.strategy == "sum"
    )

    delete_max = next(
        score
        for score in delete_scores
        if score.model_name == model_name and score.strategy == "max"
    )

    return spelling_mean, spelling_max, delete_mean, delete_max


def scores_to_table(
    spelling_scores: List[Score],
    delete_scores: List[Score],
    models: List[str],
    filename: Path,
) -> None:
    assert len(spelling_scores) == len(delete_scores)

    with open(filename, "w") as file:
        for model in models:
            file.write(
                scores_to_table_row(
                    *get_relevant_scores(model, spelling_scores, delete_scores)
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", help="write baseline table.", action="store_true",
    )
    parser.add_argument(
        "--models", help="write models table.", action="store_true",
    )
    args = parser.parse_args()

    delete_scores = disk.get_scores(disk.DELETE_SCORES_FILE)
    spelling_scores = disk.get_scores(disk.SPELLING_SCORES_FILE)

    if args.baseline:
        scores_to_table(
            spelling_scores,
            delete_scores,
            ["random", "bert_default", "bert_glue"],
            BASELINE_TABLE_TEX_FILE,
        )

    if args.models:
        scores_to_table(
            spelling_scores,
            delete_scores,
            ["bert", "roberta", "scibert"],
            TABLE_TEX_FILE,
        )
