"""
Gets the attention map for [CLS] in the final layer for every relevant sentence and saves it to disk, from where it can be loaded when needed.
"""

from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .. import disk, models, structures, tokenizers
from .. import util as base_util
from . import util


def format_special_chars(tokens: List[str]) -> List[str]:
    return [
        t.replace("Ġ", " ").replace("▁", " ").replace("</w>", "").replace("##", "")
        for t in tokens
    ]


def merge_attention_head(
    attention_head: np.ndarray,
    tokens: List[str],
    words: List[str],
    word_ends: List[str],
    verbose: int = 0,
) -> np.ndarray:
    assert attention_head.shape == (len(tokens), len(tokens)), str(attention_head.shape)

    if verbose == 1:
        print(attention_head.shape)
        print()
    if verbose == 2:
        print(attention_head)
        print()

    # step 1: merge attention *to* split words

    merged_attention = np.zeros((len(tokens), len(words)))

    for token_i, token_from in enumerate(tokens):
        attention_sum = 0
        word_j = -1
        for token_j, token_to in enumerate(tokens):
            attention_sum += attention_head[token_i, token_j]
            if token_to in word_ends[word_j + 1 :]:
                word_j = word_ends.index(token_to, word_j + 1)
                merged_attention[token_i, word_j] = attention_sum
                attention_sum = 0

    if verbose == 1:
        print(merged_attention.shape)
        print()
    if verbose == 2:
        print(merged_attention)
        print()

    final_attention = np.zeros((len(words), len(words)))

    # step 2: merge attention *from* split words

    for word_j, word in enumerate(words):
        word_i = -1
        attention_to_word = 0
        tokens_to_word_count = 0
        for token_i, token in enumerate(tokens):
            attention_to_word += merged_attention[token_i, word_j]
            tokens_to_word_count += 1

            if token in word_ends[word_i + 1 :]:
                word_i = word_ends.index(token, word_i + 1)
                attention_from_word = attention_to_word / tokens_to_word_count
                final_attention[word_i, word_j] = attention_from_word
                attention_to_word = 0
                tokens_to_word_count = 0

    if verbose == 1:
        print(final_attention.shape)
        print()
    if verbose == 2:
        print(final_attention)
        print()

    return final_attention


def get_word_ends_from_tokens(tokens: List[str], words: List[str]) -> List[str]:
    assert tokens[0] == words[0], f"{tokens[0]} != {words[0]}"
    assert tokens[-1] == words[-1]

    word_ends = []

    word_i = 0
    token_i = 0
    while word_i < len(words) and token_i < len(tokens):
        if words[word_i].lower().endswith(tokens[token_i].lower().strip()):
            word_ends.append(tokens[token_i])
            word_i += 1

        token_i += 1

    return word_ends


class ModelOutput(NamedTuple):
    words: List[str]
    attention: np.ndarray
    prediction: bool


def get_words_and_attention_and_prediction(
    sents: List[str],
    cls_model: models.SentenceClassificationModel,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
) -> List[ModelOutput]:
    if not sents:
        return []

    words = [util.tokenize_transformer_sentences(sent) for sent in sents]

    inputs = tokenizer.batch_encode_plus(
        sents, return_tensors="pt", add_special_tokens=True, padding=True
    )

    device = base_util.get_device()

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        attention = model(input_ids, output_attentions=True)[-1]

        logits = cls_model(input_ids, attn_mask=attn_mask)
        _, raw_predictions = torch.max(logits.data, 1)  # type: ignore

        predictions: List[int] = raw_predictions.tolist()

    # attention is a tuple of length 12
    # each element in attention is a tensor len(sents) x 12 x max_seq_len x max_seq_len

    results = []

    for i in range(len(sents)):
        # get tokens
        input_id_list = input_ids[i].tolist()
        tokens: List[str] = tokenizer.convert_ids_to_tokens(input_id_list)

        tokens[0] = "[CLS]"

        # last token isn't always [SEP] so we normalize
        last_token = tokens.index(tokenizer.sep_token)
        tokens[last_token] = "[SEP]"

        # only care about non-padding tokens
        tokens = tokens[: last_token + 1]
        tokens = format_special_chars(tokens)

        word_ends = get_word_ends_from_tokens(tokens, words[i])

        # get attention and remove the values that should be masked
        word_attention = []

        for attention_head in attention[11][i]:
            relevant_attn = attention_head.numpy()[: last_token + 1, : last_token + 1]
            word_attn = merge_attention_head(relevant_attn, tokens, words[i], word_ends)
            word_attention.append(word_attn)

        word_attention_arr: np.ndarray = np.stack(word_attention, axis=0)

        results.append(ModelOutput(words[i], word_attention_arr, bool(predictions[i])))

    return results


if __name__ == "__main__":
    bert_config = structures.HyperParameters(
        "./experiments/bert_base_aesw_32_1e6/params.toml"
    )
    disk_model = disk.load_finetuned_model_from_disk(
        bert_config, Path("./models-versioned/8ac1b935-200a-41fc-94a4-5d185aab1269.pt"),
    )
    if isinstance(disk_model, Exception):
        raise disk_model

    bert, _ = disk_model
    bert_tokenizer = tokenizers.get_tokenizer(bert_config)

    get_words_and_attention_and_prediction(
        ["hello world!", "My name is sam!"], bert, bert.bert.bert, bert_tokenizer
    )
