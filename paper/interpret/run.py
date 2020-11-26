import json
import pickle
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from .. import disk as base_disk
from .. import models, structures, tokenizers
from .. import util as base_util
from . import attention, disk, edits, predict, util
from .structures import Score, ScoreEncoder

# region load-models


Model = Tuple[
    models.SentenceClassificationModel, PreTrainedModel, PreTrainedTokenizerFast
]


def load_bert_default() -> Model:
    default_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    if torch.cuda.is_available():
        default_model.cuda()

    cls_model = models.SentenceClassificationModel(default_model)

    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    return cls_model, default_model.bert, bert_tokenizer


def load_bert_glue() -> Model:
    glue_model = BertForSequenceClassification.from_pretrained(
        "./models-versioned/bert_glue"
    )

    if torch.cuda.is_available():
        glue_model.cuda()

    cls_model = models.SentenceClassificationModel(glue_model)

    bert_tokenizer = BertTokenizerFast.from_pretrained("./models-versioned/bert_glue")

    return cls_model, glue_model.bert, bert_tokenizer


def load_bert() -> Model:
    bert_config = structures.HyperParameters(
        "./experiments/bert_base_aesw_32_1e6/params.toml"
    )
    disk_model = base_disk.load_finetuned_model_from_disk(
        bert_config, Path("./models-versioned/8ac1b935-200a-41fc-94a4-5d185aab1269.pt"),
    )
    if isinstance(disk_model, Exception):
        raise disk_model

    bert, _ = disk_model
    bert_tokenizer = tokenizers.get_tokenizer(bert_config)
    return bert, bert.bert.bert, bert_tokenizer


def load_scibert() -> Model:
    scibert_config = structures.HyperParameters(
        "./experiments/scibert_32_1e6/params.toml"
    )
    disk_model = base_disk.load_finetuned_model_from_disk(
        scibert_config,
        Path("./models-versioned/8128a162-5efb-4920-b63f-87a2be6fd503.pt"),
    )
    if isinstance(disk_model, Exception):
        raise disk_model

    scibert, _ = disk_model
    scibert_tokenizer = tokenizers.get_tokenizer(scibert_config)
    return scibert, scibert.bert.bert, scibert_tokenizer


def load_roberta() -> Model:
    roberta_config = structures.HyperParameters(
        "./experiments/roberta_base_aesw_32_1e6/params.toml"
    )
    disk_model = base_disk.load_finetuned_model_from_disk(
        roberta_config,
        Path("./models-versioned/503176a5-8a15-4d3b-afa0-5e093934a611.pt"),
    )
    if isinstance(disk_model, Exception):
        raise disk_model

    roberta, _ = disk_model
    roberta_tokenizer = tokenizers.get_tokenizer(roberta_config)
    return roberta, roberta.bert.roberta, roberta_tokenizer


def get_all_models(finetuned_only: bool = False) -> Dict[str, Model]:
    if finetuned_only:
        return {
            "bert": load_bert(),
            "scibert": load_scibert(),
            "roberta": load_roberta(),
        }
    else:
        return {
            "bert": load_bert(),
            "scibert": load_scibert(),
            "roberta": load_roberta(),
            "bert_glue": load_bert_glue(),
            "bert_default": load_bert_default(),
        }


# endregion


# region disk


Cache = Dict[str, Tuple[List[str], np.ndarray, bool]]


PICKLE_FOLDER = Path("./data-unversioned/attention-weights/")


def save_model_cache(model_name: str, filename: str, cache: Cache) -> None:
    with open(PICKLE_FOLDER / model_name / filename, "wb") as file:
        pickle.dump(cache, file)


def load_model_cache(model_name: str, filename: str) -> Cache:
    with open(PICKLE_FOLDER / model_name / filename, "rb") as file:
        return pickle.load(file)  # type: ignore


# endregion


# region evaluation


@dataclass
class Prediction:
    id: str
    target: Set[int]
    predictions: List[int]  # ordered but unique. From a prediction for an id

    def top_n(self, n: Optional[int] = None) -> Set[int]:
        """
        if n is not provided, use len(self.target)
        """
        if n is None:
            n = len(self.target)

        return set(self.predictions[:n])

    def jaccard_similarity(self, n: Optional[int] = None) -> float:
        """
        if n is not provided, use len(self.target)
        """
        if n is None:
            n = len(self.target)

        return len(self.target & self.top_n(n)) / len(self.target | self.top_n(n))


def cls_attn(last_layer_attn: np.ndarray) -> np.ndarray:
    heads, words_from, words_to = last_layer_attn.shape
    assert heads == 12, heads
    assert words_from == words_to

    # [:, 0, :] picks the [CLS] token's attention to all words for all heads
    return last_layer_attn[:, 0, :]


def use_strategy(
    edits: List[edits.Edit], strategy: predict.Strategy, attn_cache: Cache,
) -> List[Prediction]:
    results = []

    for edit in edits:
        if edit.id not in attn_cache:
            continue  # tokenization error
        words, attn, prediction = attn_cache[edit.id]

        assert words == edit.words, f"{words} != {edit.words}"

        best_word_predictions = predict.get_best_choices(cls_attn(attn), strategy)

        target_words = util.multi_index(words, edit.best_words)
        assert "[CLS]" not in target_words, edit
        assert "[SEP]" not in target_words, edit

        predicted_words = util.multi_index(words, best_word_predictions)
        assert "[CLS]" not in predicted_words, best_word_predictions
        assert "[SEP]" not in predicted_words, best_word_predictions

        results.append(Prediction(edit.id, edit.best_words, best_word_predictions))

    return results


def get_random_predictions(edits: List[edits.Edit]) -> List[Prediction]:
    random.seed(42)

    results = []

    for edit in edits:

        possible_indices = list(
            range(1, len(edit.words) - 1)
        )  # 1 to skip [CLS], len(edit.words)-1 to skip [SEP]
        random.shuffle(possible_indices)
        assert 0 not in possible_indices
        assert len(edit.words) - 1 not in possible_indices
        best_word_predictions = possible_indices

        target_words = util.multi_index(edit.words, edit.best_words)
        assert "[CLS]" not in target_words, edit
        assert "[SEP]" not in target_words, edit

        if not target_words:
            continue

        predicted_words = util.multi_index(edit.words, best_word_predictions)
        assert "[CLS]" not in predicted_words, best_word_predictions
        assert "[SEP]" not in predicted_words, best_word_predictions

        results.append(Prediction(edit.id, edit.best_words, best_word_predictions))

    return results


@dataclass
class EvalStrategyResult:
    exact_match_accuracy: float = 0
    jaccard_match_accuracy: float = 0
    mean_jaccard_similarity: float = 0
    top_3_match_accuracy: float = 0


def evaluate_strategy(predictions: List[Prediction]) -> EvalStrategyResult:
    total = len(predictions)
    if not total:
        return EvalStrategyResult()

    exact_matches = 0
    jaccard_sims = []
    top_3_matches = 0

    for prediction in predictions:
        if prediction.target == prediction.top_n():
            exact_matches += 1

        if prediction.target <= prediction.top_n(3):
            assert len(prediction.target) > 0
            top_3_matches += 1

        jaccard_sims.append(prediction.jaccard_similarity())

    exact_match_accuracy = exact_matches / total
    top_3_accuracy = top_3_matches / total

    jaccard_match_accuracy = len([s for s in jaccard_sims if s >= 0.5]) / total

    return EvalStrategyResult(
        exact_match_accuracy,
        jaccard_match_accuracy,
        statistics.mean(jaccard_sims),
        top_3_accuracy,
    )


# endregion


# region get-attention-weights


def get_attention_weights(
    edits: Iterable[edits.Edit],
    cls_model: models.SentenceClassificationModel,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    skip_false_predictions: bool = False,
) -> Cache:
    saved = {}

    for edit_batch in base_util.my_tqdm(base_util.chunks(iter(edits), 8)):
        sents = [edit.sent for edit in edit_batch]
        result = attention.get_words_and_attention_and_prediction(
            sents, cls_model, model, tokenizer
        )

        for edit, (words, attn, prediction) in zip(edit_batch, result):
            if skip_false_predictions and not prediction:
                continue

            if edit.words != words:
                print(edit.id)
                print(edit.words)
                print(words)
                continue

            saved[edit.id] = (words, attn, prediction)

    return saved


def save_edit_weights(all_edits: Iterable[edits.Edit], filename: str) -> None:
    all_models = get_all_models()

    # save all the weights
    for model_name, (cls_model, model, tokenizer) in all_models.items():
        cache = get_attention_weights(all_edits, cls_model, model, tokenizer)
        save_model_cache(model_name, filename, cache)


def save_comma_edit_weights() -> None:
    all_edits = edits.get_comma_edits()
    save_edit_weights(all_edits, disk.COMMA_EDIT_WEIGHTS_FILENAME)


def save_delete_edit_weights() -> None:
    all_edits = edits.get_edits_with_only_deleted_words()
    save_edit_weights(all_edits, disk.DELETE_EDIT_WEIGHTS_FILENAME)


def save_spelling_edit_weights() -> None:
    all_edits = edits.get_edits_with_only_spelling_errors(again=True)
    save_edit_weights(all_edits, disk.SPELLING_EDIT_WEIGHTS_FILENAME)


# endregion


# region evaluation


def get_universal_correct_predictions(weights_filename: str) -> Set[str]:
    bert_correct_predictions = set()

    bert_cache = load_model_cache("bert", weights_filename)
    for sent_id, (_, _, prediction) in bert_cache.items():
        if prediction:
            bert_correct_predictions.add(sent_id)

    roberta_correct_predictions = set()

    roberta_cache = load_model_cache("roberta", weights_filename)
    for sent_id, (_, _, prediction) in roberta_cache.items():
        if prediction:
            roberta_correct_predictions.add(sent_id)

    scibert_correct_predictions = set()

    scibert_cache = load_model_cache("scibert", weights_filename)
    for sent_id, (_, _, prediction) in scibert_cache.items():
        if prediction:
            scibert_correct_predictions.add(sent_id)

    universal = (
        bert_correct_predictions
        & scibert_correct_predictions
        & roberta_correct_predictions
    )

    assert len(universal) <= len(bert_correct_predictions)
    assert len(universal) <= len(scibert_correct_predictions)
    assert len(universal) <= len(roberta_correct_predictions)

    return universal


def evaluate_all_strategies_on_comma_edits() -> None:
    comma_edits = edits.get_comma_edits()

    all_strategies: List[predict.Strategy] = ["sum", "max"]

    skip_false_predictions_models = [
        "bert",
        "scibert",
        "roberta",
        "bert_default",
        "bert_glue",
    ]

    scores: List[Score] = []

    universal = get_universal_correct_predictions(disk.COMMA_EDIT_WEIGHTS_FILENAME)

    for model_name in skip_false_predictions_models:
        # load the weights for a model
        cache = load_model_cache(model_name, disk.COMMA_EDIT_WEIGHTS_FILENAME)

        # only do predictions that all models get
        cache = {key: value for key, value in cache.items() if key in universal}

        # for each strategy and type of edit, make a bunch of guesses
        for strategy in all_strategies:

            predictions = use_strategy(comma_edits, strategy, cache)

            result = evaluate_strategy(predictions)

            scores.append(
                Score(
                    model_name,
                    strategy,
                    result.exact_match_accuracy,
                    result.jaccard_match_accuracy,
                    result.mean_jaccard_similarity,
                    result.top_3_match_accuracy,
                    False,
                )
            )

    random_predictions = get_random_predictions(comma_edits)

    result = evaluate_strategy(random_predictions)

    scores.append(
        Score(
            "random",
            "sum",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    scores.append(
        Score(
            "random",
            "max",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    with open(
        "./data-unversioned/attention-weights/scores/comma-edit.json", "w"
    ) as file:
        json.dump(scores, file, indent=4, cls=ScoreEncoder)


def evaluate_all_strategies_on_spelling_edits() -> None:
    spelling_edits = edits.get_edits_with_only_spelling_errors()

    all_strategies: List[predict.Strategy] = ["sum", "max"]

    skip_false_predictions_models = [
        "bert",
        "scibert",
        "roberta",
        "bert_default",
        "bert_glue",
    ]

    scores: List[Score] = []

    universal = get_universal_correct_predictions(disk.SPELLING_EDIT_WEIGHTS_FILENAME)

    for model_name in skip_false_predictions_models:
        # load the weights for a model
        cache = load_model_cache(model_name, disk.SPELLING_EDIT_WEIGHTS_FILENAME)

        # only do predictions that all models get
        cache = {key: value for key, value in cache.items() if key in universal}

        # for each strategy and type of edit, make a bunch of guesses
        for strategy in all_strategies:

            predictions = use_strategy(spelling_edits, strategy, cache,)

            result = evaluate_strategy(predictions)

            scores.append(
                Score(
                    model_name,
                    strategy,
                    result.exact_match_accuracy,
                    result.jaccard_match_accuracy,
                    result.mean_jaccard_similarity,
                    result.top_3_match_accuracy,
                    False,
                )
            )

    random_predictions = get_random_predictions(spelling_edits)

    result = evaluate_strategy(random_predictions)

    scores.append(
        Score(
            "random",
            "sum",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    scores.append(
        Score(
            "random",
            "max",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    with open(
        "./data-unversioned/attention-weights/scores/spelling-edit.json", "w"
    ) as file:
        json.dump(scores, file, indent=4, cls=ScoreEncoder)


def evaluate_all_strategies_on_delete_edits() -> None:
    delete_edits = edits.get_edits_with_only_deleted_words()

    all_strategies: List[predict.Strategy] = ["sum", "max"]

    skip_false_predictions_models = [
        "bert",
        "scibert",
        "roberta",
        "bert_default",
        "bert_glue",
    ]

    scores: List[Score] = []

    universal = get_universal_correct_predictions(disk.DELETE_EDIT_WEIGHTS_FILENAME)

    for model_name in skip_false_predictions_models:
        # load the weights for a model
        cache = load_model_cache(model_name, disk.DELETE_EDIT_WEIGHTS_FILENAME)

        # only do predictions that all models get
        cache = {key: value for key, value in cache.items() if key in universal}

        # for each strategy and type of edit, make a bunch of guesses
        for strategy in all_strategies:

            predictions = use_strategy(delete_edits, strategy, cache)

            result = evaluate_strategy(predictions)

            scores.append(
                Score(
                    model_name,
                    strategy,
                    result.exact_match_accuracy,
                    result.jaccard_match_accuracy,
                    result.mean_jaccard_similarity,
                    result.top_3_match_accuracy,
                    False,
                )
            )

    random_predictions = get_random_predictions(delete_edits)

    result = evaluate_strategy(random_predictions)

    scores.append(
        Score(
            "random",
            "sum",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    scores.append(
        Score(
            "random",
            "max",
            result.exact_match_accuracy,
            result.jaccard_match_accuracy,
            result.mean_jaccard_similarity,
            result.top_3_match_accuracy,
            False,
        )
    )

    with open(
        "./data-unversioned/attention-weights/scores/delete-edit.json", "w"
    ) as file:
        json.dump(scores, file, indent=4, cls=ScoreEncoder)


# endregion

if __name__ == "__main__":
    print("run 'python -m paper.interpret'")
