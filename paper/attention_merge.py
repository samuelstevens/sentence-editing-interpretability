from typing import List

import numpy as np


def merge_attention_head(
    attention_head: np.ndarray,
    tokens: List[str],
    words: List[str],
    word_ends: List[str],
    verbose: int = 0,
) -> np.ndarray:
    assert attention.shape == (len(tokens), len(tokens))

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


tokens = ["[CLS]", " time", "-", "v", "ary", "ing", "[SEP]"]
words = ["[CLS]", "time-varying", "[SEP]"]
word_ends = ["[CLS]", "ing", "[SEP]"]
attention = np.ones((len(tokens), len(tokens)))
print(merge_attention_head(attention, tokens, words, word_ends, verbose=0))

tokens = ["straw", "##berries"]
words = ["strawberries"]
word_ends = ["##berries"]
attention = np.array([[0.2, 0.8], [0.8, 0.2]])
print(merge_attention_head(attention, tokens, words, word_ends, verbose=0))

tokens = ["straw", "##berries"]
words = ["strawberries"]
word_ends = ["##berries"]
attention = np.array([[0.2, 0.8], [0.2, 0.8]])
print(merge_attention_head(attention, tokens, words, word_ends, verbose=0))

tokens = ["and", "and"]
words = ["and", "and"]
word_ends = ["and", "and"]
attention = np.array([[0.9, 0.1], [0.1, 0.9]])
print(merge_attention_head(attention, tokens, words, word_ends, verbose=0))
