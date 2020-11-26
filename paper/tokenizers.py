from typing import List, Set, Tuple, cast

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .structures import HyperParameters

AESW_TOKENS = ["_MATH_", "_REF_", "_MATHDISP_", "_CITE_"]
AESW_IDS: List[int] = []
BERT_IDS: Set[int] = set()
MAX_LEN = 256


def get_tokenizer(config: HyperParameters) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, do_lower_case=True, use_fast=True
    )

    tokenizer.add_special_tokens({"additional_special_tokens": AESW_TOKENS})

    global AESW_IDS
    AESW_IDS = tokenizer.convert_tokens_to_ids(AESW_TOKENS)

    global BERT_IDS
    BERT_IDS = set(tokenizer.all_special_ids) - set(AESW_IDS)

    return tokenizer


def decode(ids: List[int], tokenizer: PreTrainedTokenizerFast) -> str:
    ids = list(filter(lambda i: i not in BERT_IDS, ids))

    return cast(str, tokenizer.decode(ids, skip_special_tokens=False))


def encode_batch(
    sentences: List[str], tokenizer: PreTrainedTokenizerFast,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Tokenizes and pads sentences to the max length in the list of sentences. Returns tensors for input ids and tensors for attention masking.
    """
    input_ids: List[List[int]] = []

    for text in sentences:
        tokenized = tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=MAX_LEN,  # Do truncate to MAX_LEN
            truncation=True,  # Do truncate
            padding=False,  # Don't pad
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids.append(tokenized["input_ids"])

    max_len = max([len(i) for i in input_ids])

    padded_input_ids: List[Tensor] = []
    attention_masks: List[Tensor] = []

    for sent in input_ids:
        num_pads = max_len - len(sent)

        padded_input_ids.append(
            torch.tensor(sent + [tokenizer.pad_token_id] * num_pads)  # type: ignore
        )
        attention_masks.append(torch.tensor([1] * len(sent) + [0] * num_pads))  # type: ignore

    return padded_input_ids, attention_masks
