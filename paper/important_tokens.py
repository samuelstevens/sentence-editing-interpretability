from pathlib import Path
from typing import List

from lxml import etree
from tqdm import tqdm

from . import aesw_to_sentences


def deleted_character_mask(sent_elem: etree._Element) -> List[bool]:
    assert sent_elem.tag == "sentence"

    mask = [False] * (len(str(sent_elem.text)) if sent_elem.text else 0)

    for del_ins in sent_elem:
        if del_ins.tag == "del" and del_ins.text:
            mask.extend([True] * len(str(del_ins.text)))
        if del_ins.tail:
            mask.extend([False] * len(str(del_ins.tail)))

    return mask


def surrounding_inserted_character_mask(sent_elem: etree._Element) -> List[bool]:
    assert sent_elem.tag == "sentence"

    mask = [False] * (len(str(sent_elem.text)) if sent_elem.text else 0)

    in_ins = False

    for del_ins in sent_elem:
        if del_ins.tag == "del":
            in_ins = False
            mask.extend([False] * len(str(del_ins.text)))
        else:
            assert not in_ins, "nested <ins>"
            in_ins = True
            if len(mask) > 0:
                mask[-1] = True
            if len(mask) > 1:
                mask[-2] = True
        if del_ins.tail:
            mask.extend([False] * len(str(del_ins.tail)))
            if in_ins:
                mask[-len(str(del_ins.tail))] = True
                in_ins = False

    return mask


def test_masks() -> None:
    xml_filepath = Path("./data-unversioned/aesw/aesw2016(v1.2)_dev.xml")

    for event, sent_elem in tqdm(etree.iterparse(str(xml_filepath), tag="sentence")):
        sent_id = aesw_to_sentences.extract_sentence_id(sent_elem)
        mask = deleted_character_mask(sent_elem)
        assert len(mask) == len(aesw_to_sentences.extract_sentence(sent_elem)), sent_id
        try:
            mask = surrounding_inserted_character_mask(sent_elem)
            assert len(mask) == len(
                aesw_to_sentences.extract_sentence(sent_elem)
            ), sent_id
        except AssertionError as err:
            print(sent_id, err)
            raise
        except IndexError as err:
            raise ValueError(sent_id, err)


def merge_char_masks(mask1: List[bool], mask2: List[bool]) -> List[bool]:
    assert len(mask1) == len(mask2)
    return [a or b for a, b in zip(mask1, mask2)]


def get_token_mask(tokens: List[str], important_char_mask: List[bool]) -> List[bool]:
    ch_idx_to_tok_idx = []

    for tok_idx, token in enumerate(tokens):
        ch_idx_to_tok_idx.extend([tok_idx] * len(token))

    assert len(ch_idx_to_tok_idx) == len(
        important_char_mask
    ), f"{len(ch_idx_to_tok_idx)} != {len(important_char_mask)}"

    important_token_indices = set()
    for tok_idx, ch_mask in zip(ch_idx_to_tok_idx, important_char_mask):
        if ch_mask:
            important_token_indices.add(tok_idx)

    token_mask = []
    for tok_idx, token in enumerate(tokens):
        if tok_idx in important_token_indices:
            token_mask.append(True)
        else:
            token_mask.append(False)

    return token_mask


def show_token_mask(tokens: List[str], token_mask: List[bool]) -> None:
    assert len(tokens) == len(token_mask)
    sent_builder = []
    mask_builder = []
    for tok, mask in zip(tokens, token_mask):
        sent_builder.append(tok)
        if mask:
            # print(f"'{tok}'", len(tok))
            mask_builder.append("".join(["*"] * len(tok)))
        else:
            mask_builder.append("".join([" "] * len(tok)))

    assert len(mask_builder) == len(
        sent_builder
    ), f"{len(mask_builder)} != {len(sent_builder)}"

    print(" ".join(sent_builder))
    print(" ".join(mask_builder))


if __name__ == "__main__":
    elem = etree.fromstring(
        """<sentence sid="33.10">The apparent propagation speeds are 83.5_MATH_1.8 km s_MATH_ and 100.5_MATH_4.2 km s_MATH_ in 171 Å and 193 Å channels, respectively. </sentence>"""
    )

    tokens = [
        "The",
        " apparent",
        " propagation",
        " speeds",
        " are",
        " 83",
        ".",
        "5",
        "_MATH_",
        "1",
        ".",
        "8",
        " km",
        " s",
        "_MATH_",
        " and",
        " 100",
        ".",
        "5",
        "_MATH_",
        "4",
        ".",
        "2",
        " km",
        " s",
        "_MATH_",
        " in",
        " 171",
        " Ã",
        " and",
        " 193",
        " Ã",
        " channels",
        ",",
        " respectively",
        ".",
        " ",
    ]

    mask = deleted_character_mask(elem)
    assert len(mask) == len(aesw_to_sentences.extract_sentence(elem))

    mask = surrounding_inserted_character_mask(elem)
    assert len(mask) == len(aesw_to_sentences.extract_sentence(elem))

    print(len(aesw_to_sentences.extract_sentence(elem)))

    mask = merge_char_masks(
        deleted_character_mask(elem), surrounding_inserted_character_mask(elem)
    )

    token_mask = get_token_mask(tokens, mask)

    show_token_mask(tokens, token_mask)
