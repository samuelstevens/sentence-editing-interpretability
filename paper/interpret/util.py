from typing import Iterable, List, TypeVar

import nltk

T = TypeVar("T")


def tokenize_transformer_sentences(
    sent: str, add_special_tokens: bool = True
) -> List[str]:
    if add_special_tokens:
        sent = "[CLS] " + sent + " [SEP]"

    raw_tokens = nltk.word_tokenize(sent)

    def fix_quotes(token: str) -> str:
        if token == "``":
            return '"'
        if token == "''":
            return '"'
        return token

    raw_tokens = [fix_quotes(token) for token in raw_tokens]

    arbitrary_token = "\t"

    return (
        arbitrary_token.join(raw_tokens)
        .replace(f"[{arbitrary_token}CLS{arbitrary_token}]", "[CLS]")
        .replace(f"[{arbitrary_token}SEP{arbitrary_token}]", "[SEP]")
        .split()
    )


def flatten(nested_list: List[List[T]]) -> List[T]:
    """
    Takes a list of lists and returns a flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def multi_index(lst: List[T], indices: Iterable[int]) -> List[T]:
    result = []
    for i in indices:
        result.append(lst[i])
    return result
