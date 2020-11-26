"""
Given a sentence, an important word index, and an attention map, use one of 14 strategies to find the best word.
"""

from typing import List

import numpy as np
from typing_extensions import Literal

Strategy = Literal["sum", "max"]


def get_best_choices(attn: np.ndarray, strategy: Strategy) -> List[int]:
    heads, words = attn.shape
    assert heads == 12

    if strategy == "sum":
        attn = attn.sum(axis=0)
    elif strategy == "max":
        attn = attn.max(axis=0)  # prevents picking the same word twice
    else:
        raise ValueError(strategy)

    assert len(attn) == words, f"should be {words} different options; got {len(attn)}"

    indices = attn.argsort()[::-1]  # gets indices from max attention to min attention
    best_choices: List[int] = indices.tolist()

    # filter first and last word because it's [CLS] and [SEP]
    best_choices.remove(0)  # [CLS]
    best_choices.remove(max(best_choices))  # [SEP]

    return best_choices
