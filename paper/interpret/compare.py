import argparse
import sys
from typing import Any, List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .. import models
from . import attention, run, util


def plot_cls_attention(
    tokens: List[str], attn: np.ndarray, title: str, word_ticks: bool, ax: Any = None
) -> Any:
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    heads = list(range(12))

    if not ax:
        ax = plt.gca()

    im = ax.imshow(
        attn.T, cmap="Reds", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    )

    ax.set_xticks(np.arange(len(heads)))
    ax.set_xticklabels([])
    ax.set_xlabel("Attention Heads")

    if word_ticks:
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens, size=12)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_title(title)
    return im


def plot_word_level_attention(
    sent: str,
    cls_model: models.SentenceClassificationModel,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    title: str,
    word_ticks: bool,
    **plotkwargs: Any,
) -> Any:
    output = attention.get_words_and_attention_and_prediction(
        [sent], cls_model, model, tokenizer
    )[0]

    title += f" ({'needs edit' if output.prediction else 'no edit'})"

    return plot_cls_attention(
        output.words, run.cls_attn(output.attention), title, word_ticks, **plotkwargs
    )


def plot_finetuned_models_comparison(sentence: str) -> None:
    height = 2 + 0.2 * len(util.tokenize_transformer_sentences(sentence))
    fig, axes = plt.subplots(1, 2, figsize=(12, height), constrained_layout=True)

    ((ax1, ax2)) = axes

    bert_default_cls, bert_default, bert_default_tokenizer = run.load_bert_default()

    im = plot_word_level_attention(
        sentence,
        bert_default_cls,
        bert_default,
        bert_default_tokenizer,
        title="Before",
        word_ticks=True,
        ax=ax1,
    )

    bert_cls, bert, bert_tokenizer = run.load_bert()

    im = plot_word_level_attention(
        sentence,
        bert_cls,
        bert,
        bert_tokenizer,
        word_ticks=False,
        title="After",
        ax=ax2,
    )

    # fig.tight_layout()

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        aspect=30,
        use_gridspec=True,
    )
    cbar.ax.set_xlabel("Attention", va="center")


def plot_all_word_models(sentence: str) -> None:
    height = 2 + 0.2 * len(util.tokenize_transformer_sentences(sentence))
    fig, axes = plt.subplots(1, 3, figsize=(12, height), constrained_layout=True)

    ((ax1, ax2, ax3)) = axes

    bert_cls, bert, bert_tokenizer = run.load_bert()

    im = plot_word_level_attention(
        sentence, bert_cls, bert, bert_tokenizer, title="BERT", word_ticks=True, ax=ax1
    )

    scibert_cls, scibert, scibert_tokenizer = run.load_scibert()

    im = plot_word_level_attention(
        sentence,
        scibert_cls,
        scibert,
        scibert_tokenizer,
        title="SciBERT",
        word_ticks=False,
        ax=ax2,
    )

    roberta_cls, roberta, roberta_tokenizer = run.load_roberta()

    im = plot_word_level_attention(
        sentence,
        roberta_cls,
        roberta,
        roberta_tokenizer,
        title="RoBERTa",
        word_ticks=False,
        ax=ax3,
    )

    # fig.tight_layout()

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        aspect=30,
        use_gridspec=True,
    )
    cbar.ax.set_xlabel("Attention", va="center")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        help="filename to save the file to. If not provided, then show chart interactively",
        type=str,
    )
    parser.add_argument(
        "--all-models",
        help="Whether to compare all three models (BERT, RoBERTa and SciBERT) on the sentence",
        action="store_true",
    )
    parser.add_argument(
        "--finetuning",
        help="Whether to compare the effects of fine-tuning (BERT before and after fine-tuning) on the sentence",
        action="store_true",
    )
    parser.add_argument(
        "sentence", help="sentence to compare", type=str,
    )
    args = parser.parse_args()

    if args.all_models and args.finetuning:
        print("Can only make a graph for finetuning OR all models at once.")
        sys.exit(1)

    if args.all_models:
        plot_all_word_models(args.sentence)
        if args.filename is None:
            plt.show()
        else:
            plt.savefig(args.filename)
        sys.exit(0)

    if args.finetuning:
        plot_finetuned_models_comparison(args.sentence)
        if args.filename is None:
            plt.show()
        else:
            plt.savefig(args.filename)
        sys.exit(0)

"""
python -m paper.interpret.compare --all-models --filename "./docs/images/saturn's.pdf" "This allows us to observe Saturn's moons."
python -m paper.interpret.compare --all-models --filename "./docs/images/we'll.pdf" "(We'll represent a signature as an encrypted message digest):"
python -m paper.interpret.compare --finetuning --filename "./docs/images/bert.pdf" "The algorithm descripted in the previous sections has several advantages."
"""
