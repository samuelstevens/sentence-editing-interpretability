"""
Find the correct edits. Also find the best word "mask": an index for the most important word according to the annotated data.
"""
import pickle
import string
from typing import List, NamedTuple, Set, Tuple

from lxml import etree
from spellchecker import SpellChecker

from .. import aesw_to_sentences
from .. import util as base_util
from . import disk, util

spell = SpellChecker()


class Edit(NamedTuple):
    id: str
    sent: str
    words: List[str]
    best_words: Set[int]


def is_single_comma_deleted(sent_elem: etree._Element) -> bool:
    assert sent_elem.tag == "sentence"

    if not is_only_deleted(sent_elem):
        return False

    commas_deleted = 0
    is_something_else = False

    for edit_elem in sent_elem:
        if edit_elem.tag == "del":
            if edit_elem.text == ",":
                commas_deleted += 1
            else:
                is_something_else = True

    return commas_deleted == 1 and not is_something_else


def is_only_deleted(sent_elem: etree._Element) -> bool:
    assert sent_elem.tag == "sentence"

    if len(sent_elem) == 0:
        return False

    for edit_elem in sent_elem:
        if edit_elem.tag != "del":
            return False

    return True


def is_only_spelling_error(sent_elem: etree._Element) -> bool:
    assert sent_elem.tag == "sentence"

    if len(sent_elem) != 2:  # should be one delete, one insert
        return False

    del_elem, ins_elem = list(sent_elem)

    if (
        del_elem.tail
    ):  # should be no text between the deleted text and the inserted text
        return False

    if del_elem.tag != "del" or ins_elem.tag != "ins":
        return False

    misspelled_words = spell.unknown([del_elem.text])

    if not misspelled_words:
        return False

    misspelled_word = base_util.get(misspelled_words)

    if misspelled_word in string.whitespace or " " in misspelled_word:
        return False

    corrections = spell.unknown([ins_elem.text])

    if (
        len(corrections) > 0
    ):  # still misspelled, but not always correct. Might want to use edit distance
        return False

    return True


def get_deleted_word_indices(sent_elem: etree._Element) -> Tuple[List[str], Set[int]]:
    assert sent_elem.tag == "sentence"

    sent_text, mask = extract_sentence_with_mask(sent_elem)
    words = util.tokenize_transformer_sentences(sent_text, add_special_tokens=False)

    if not any([elem.tag == "del" for elem in sent_elem]):
        return words, set()

    word_i = 0
    word_ch_i = 0
    sent_i = 0

    best_words = set()

    while word_i < len(words) and sent_i < len(sent_text):
        if mask[sent_i] != " ":
            best_words.add(word_i)
        word_ch_i += 1
        sent_i += 1

        if word_ch_i >= len(words[word_i]):
            word_i += 1
            word_ch_i = 0
            # skip whitespace in sent
            while sent_i < len(sent_text) and sent_text[sent_i] in string.whitespace:
                sent_i += 1

        if sent_i >= len(sent_text):
            break

        assert (
            words[word_i][word_ch_i] == sent_text[sent_i]
        ), f"'{words[word_i][word_ch_i]}' != '{sent_text[sent_i]}' ({aesw_to_sentences.extract_sentence_id(sent_elem)})"

    words.insert(0, "[CLS]")
    words.append("[SEP]")
    best_words = set([b + 1 for b in best_words])  # all incremented because of [CLS]

    return words, best_words


def extract_sentence_with_mask(sent_elem: etree._Element) -> Tuple[str, str]:
    assert sent_elem.tag == "sentence"
    string_builder = [str(sent_elem.text) if sent_elem.text else ""]
    mask_builder = [" " * len(str(sent_elem.text)) if sent_elem.text else ""]

    for del_ins in sent_elem:
        if del_ins.tag == "del" and del_ins.text:
            string_builder.append(str(del_ins.text))
            mask_builder.append("*" * len(string_builder[-1]))
        if del_ins.tail:
            string_builder.append(str(del_ins.tail))
            mask_builder.append(" " * len(string_builder[-1]))

    assert len(mask_builder) == len(string_builder)

    sent = "".join(string_builder)
    mask = "".join(mask_builder)

    assert len(sent) == len(mask)

    return sent, mask


def save_edits(edits: List[Edit], filename: str) -> None:
    with open(disk.INTERPRETABILITY_FOLDER / filename, "wb") as file:
        pickle.dump(edits, file)


def load_edits(filename: str) -> List[Edit]:
    with open(disk.INTERPRETABILITY_FOLDER / filename, "rb") as file:
        return pickle.load(file)  # type: ignore


def get_comma_edits(again: bool = False) -> List[Edit]:
    filename = "comma.pckl"
    try:
        if not again:
            return load_edits(filename)
    except FileNotFoundError:
        pass

    results = []

    for _, sent_elem in base_util.my_tqdm(
        etree.iterparse(disk.DEV_XML, tag="sentence")
    ):
        sent_id = aesw_to_sentences.extract_sentence_id(sent_elem)
        sent = aesw_to_sentences.extract_sentence(sent_elem)

        if not is_single_comma_deleted(sent_elem):
            continue

        words, best_words = get_deleted_word_indices(sent_elem)

        edit = Edit(sent_id, sent, words, best_words)

        results.append(edit)

    save_edits(results, filename)

    return results


def get_edits_with_only_deleted_words(again: bool = False) -> List[Edit]:
    filename = "deleted.pckl"
    try:
        if not again:
            return load_edits(filename)
    except FileNotFoundError:
        pass

    results = []

    for _, sent_elem in base_util.my_tqdm(
        etree.iterparse(disk.DEV_XML, tag="sentence")
    ):
        sent_id = aesw_to_sentences.extract_sentence_id(sent_elem)
        sent = aesw_to_sentences.extract_sentence(sent_elem)

        if not is_only_deleted(sent_elem):
            continue

        words, best_words = get_deleted_word_indices(sent_elem)

        edit = Edit(sent_id, sent, words, best_words)

        results.append(edit)

    save_edits(results, filename)

    return results


def get_edits_with_only_spelling_errors(again: bool = False) -> List[Edit]:
    filename = "spelling.pckl"
    try:
        if not again:
            return load_edits(filename)
    except FileNotFoundError:
        pass

    results = []

    for _, sent_elem in base_util.my_tqdm(
        etree.iterparse(disk.DEV_XML, tag="sentence")
    ):
        sent_id = aesw_to_sentences.extract_sentence_id(sent_elem)
        sent = aesw_to_sentences.extract_sentence(sent_elem)

        if not is_only_spelling_error(sent_elem):
            continue

        words, best_words = get_deleted_word_indices(sent_elem)

        edit = Edit(sent_id, sent, words, best_words)

        results.append(edit)

    save_edits(results, filename)

    return results


# region testing


def test_comma_edits() -> None:
    e = etree.fromstring(
        '<sentence sid="250.9">This, which<del>,</del> is difficult, can be identified trivially</sentence>'
    )

    words, best_words = get_deleted_word_indices(e)

    assert words == [
        "[CLS]",
        "This",
        ",",
        "which",
        ",",
        "is",
        "difficult",
        ",",
        "can",
        "be",
        "identified",
        "trivially",
        "[SEP]",
    ], words
    assert best_words == {4}, best_words

    assert is_single_comma_deleted(e)

    e = etree.fromstring(
        '<sentence sid="250.9">Then<del>,</del> the SVM responses can be computed respectively for the new positive test samples (dots) and the negative ones (stars). </sentence>'
    )

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    assert words[best_word] == ",", words[best_word]

    assert is_single_comma_deleted(e)

    e = etree.fromstring(
        '<sentence sid="250.9">Then the SVM <del>responses</del> can be computed respectively for the new positive test samples (dots) and the negative ones (stars). </sentence>'
    )

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    assert words[best_word] == "responses"

    e = etree.fromstring(
        '<sentence sid="250.9">Then the SVM can be computed respectively for <del>the</del> new positive test samples (dots) and the negative ones (stars). </sentence>'
    )

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    assert words[best_word] == "the"
    assert best_word == 9, best_word

    e = etree.fromstring(
        '<sentence sid="136.1">The same reasoning as above yields a probability measure _MATH_, in particular _MATH_ on _MATH_, and an optional measure _MATH_ such that _MATH_ _MATH_-a.s. for all _MATH_, i.e.<del>,</del> _MATH_ for _MATH_, and _REF_ holds.</sentence>'
    )

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    assert words[best_word] == ","

    assert is_single_comma_deleted(e)

    e = etree.fromstring(
        '<sentence sid="2.0">Neville elimination is an alternative procedure to that of Gauss to transform a square matrix _MATH_ into an upper<del> </del><ins>-</ins>triangular one.</sentence>'
    )

    sent, mask = extract_sentence_with_mask(e)

    words, best_words = get_deleted_word_indices(e)

    assert best_words == set()  # best words is empty because a space was deleted
    assert not is_single_comma_deleted(e)

    e = etree.fromstring(
        '<sentence sid="195.0">Now it is a routine matter to check that indeed _MATH_, where _MATH_, and where _MATH_ is the expansion of _MATH_ with all Skolem functions _MATH_ (including the "Skolem constants", i.e. witnesses). (i) follows immediately from (ii): <del>Let</del><ins>let</ins> _MATH_ (with _MATH_), then there is an _MATH_ such that _MATH_. </sentence>'
    )

    sent, mask = extract_sentence_with_mask(e)

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    e = etree.fromstring(
        '<sentence sid="2178.3">Among these parameters are the vertical gas velocity<del>,</del> v<del>,</del> and the line width w. </sentence>'
    )

    sent, mask = extract_sentence_with_mask(e)

    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    e = etree.fromstring(
        '<sentence sid="32.2">Elements at locations<del>,</del> _MATH_, with fluxes<del>,</del> _MATH_, will create a vertical field _MATHDISP_<del>,</del> in the photospheric plane. </sentence>'
    )

    sent, mask = extract_sentence_with_mask(e)
    words, best_words = get_deleted_word_indices(e)
    best_word = base_util.get(best_words)

    assert not is_single_comma_deleted(e)


def test_deleted_word_edits() -> None:
    e = etree.fromstring(
        '<sentence sid="26755.3">Thus as _MATH_, we get <del>that </del>_MATH_, which means that _MATH_. </sentence>'
    )

    assert is_only_deleted(e)
    words, best_words = get_deleted_word_indices(e)
    assert len(words) == 16  # 14 + 2 for [CLS] and [SEP]
    assert best_words == {7}

    e = etree.fromstring(
        '<sentence sid="5.1">To do this, we consider a full-order Luenberger state observer _CITE_, _CITE_ given by<del>:</del> _MATHDISP_ where _MATH_ is the estimate of the state and _MATH_ is the observer gain and defines the estimation error dynamics. </sentence>'
    )

    assert is_only_deleted(e)
    words, best_words = get_deleted_word_indices(e)
    assert len(words) == 41, words
    assert best_words == {17}, f"{words} {best_words}"

    e = etree.fromstring(
        '<sentence sid="32.2">Elements at locations<del>,</del> _MATH_, with fluxes<del>,</del> _MATH_, will create a vertical field _MATHDISP_<del>,</del> in the photospheric plane. </sentence>'
    )

    assert is_only_deleted(e)
    words, best_words = get_deleted_word_indices(e)
    assert len(words) == 25, f"{words} {len(words)}"
    assert best_words == {4, 9, 18}, f"{words} {best_words}"


# endregion

if __name__ == "__main__":
    test_deleted_word_edits()
    test_comma_edits()

    # e = etree.fromstring(
    #     '<sentence sid="33.9">Recently, _CITE_ also <del>find</del><ins>found</ins> that velocities of the propagating disturbances in structures located at sunspot regions are temperature dependent. </sentence>'
    # )

    # get_edits_with_only_spelling_errors(again=True)
