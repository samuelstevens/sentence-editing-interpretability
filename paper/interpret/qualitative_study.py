import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lxml import etree
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .. import aesw_to_sentences, models
from .. import util as base_util
from . import attention, disk, predict, run, util

SPELLING_ERRORS = [
    "22702.0",
    "25201.6",
    "31220.0",
    "17801.1",
    "25891.0",
    "4621.0",
    "8425.2",
    "17235.6",
    "12638.1",
    "23326.0",
    "11808.0",
    "28425.1",
    "19763.4",
    "20660.4",
    "22293.2",
    "25213.0",
    "16993.1",
    "30810.2",
    "13639.1",
    "23442.0",
    "28347.0",
    "20614.2",
    "9504.4",
    "11752.8",
    "2720.0",
]

DELETED_ERRORS = [
    "21957.0",
    "20697.2",
    "7945.0",
    "3955.1",
    "17863.3",
    "31349.5",
    "9986.2",
    "9365.6",
    "11317.1",
    "16734.1",
    "26160.4",
    "24697.0",
    "31330.1",
    "4099.1",
    "19995.0",
    "1201.1",
    "23383.2",
    "23731.1",
    "2494.6",
    "15589.0",
    "13069.4",
    "17227.1",
    "924.2",
    "4175.5",
    "16270.1",
]


@dataclass
class QualitativeEdit:
    sent_id: str
    sent_elem: etree._Element
    words: List[str]
    top_3_words: Dict[str, Tuple[str, str, str]]
    predictions: Dict[str, bool]
    target: Optional[bool] = None


def get_edits(ids: List[str]) -> List[QualitativeEdit]:
    results = []

    id_set = set(ids)

    for _, sent_elem in base_util.my_tqdm(
        etree.iterparse(disk.DEV_XML, tag="sentence")
    ):
        sent_id = aesw_to_sentences.extract_sentence_id(sent_elem)
        if sent_id not in id_set:
            continue

        words = util.tokenize_transformer_sentences(
            aesw_to_sentences.extract_sentence(sent_elem), add_special_tokens=True
        )
        results.append(QualitativeEdit(sent_id, sent_elem, words, {}, {}))

    random.seed(42)
    random.shuffle(results)

    return results


def escape_special_tokens(dirty: str) -> str:
    return (
        dirty.replace("_MATH_", r"\_MATH\_")
        .replace("_CITE_", r"\_CITE\_")
        .replace("_MATHDISP_", r"\_MATHDISP\_")
        .replace("_REF_", r"\_REF\_")
    )


def to_markdown_row(edit: QualitativeEdit) -> str:
    string_builder = []

    string_builder.append(f"({edit.sent_id}) ")

    if edit.sent_elem.text:
        string_builder.append(str(edit.sent_elem.text))

    for edit_elem in edit.sent_elem:
        if edit_elem.tag == "del":
            string_builder.append(f"~~{str(edit_elem.text).strip()}~~")
            if str(edit_elem.text)[-1] == " ":
                string_builder.append(" ")
        elif edit_elem.tag == "ins":
            string_builder.append(f"**{str(edit_elem.text).strip()}**")
            if str(edit_elem.text)[-1] == " ":
                string_builder.append(" ")
        else:
            raise ValueError(str(edit_elem))

        if edit_elem.tail:
            string_builder.append(str(edit_elem.tail))

    string_builder.append("\n\n")

    string_builder.append("|   | bert | scibert | roberta |\n")
    string_builder.append("|---|---|---|---|\n")
    for i in range(3):
        string_builder.append(
            f"|{i+1}. | {edit.top_3_words['bert'][i]} | {edit.top_3_words['scibert'][i]} | {edit.top_3_words['roberta'][i]} |\n"
        )

    # string_builder.append(
    #     f"| Correct prediction | {edit.predictions['bert'] == needs_edit} | {edit.predictions['scibert']== needs_edit} | {edit.predictions['roberta']== needs_edit} |\n"
    # )
    string_builder.append("|Relevant?| | | |\n\n")

    return escape_special_tokens("".join(string_builder))


def set_top_3_words(
    edits: List[QualitativeEdit],
    model_name: str,
    cls_model: models.SentenceClassificationModel,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    for edit in iter(edits):
        sent = aesw_to_sentences.extract_sentence(edit.sent_elem)

        words, attn, prediction = attention.get_words_and_attention_and_prediction(
            [sent], cls_model, model, tokenizer
        )[0]

        if edit.words != words:
            print(edit.sent_id)
            print(edit.words)
            print(words)
            continue

        top_3_indices = predict.get_best_choices(run.cls_attn(attn), "sum")[:3]

        top_3_words = [edit.words[i] for i in top_3_indices]

        # print(edit.sent_id, model_name, top_3_indices, top_3_words)

        assert len(top_3_words) == 3

        edit.top_3_words[model_name] = (
            top_3_words[0],
            top_3_words[1],
            top_3_words[2],
        )

        edit.predictions[model_name] = prediction
        edit.target = len(edit.sent_elem) > 0

        assert prediction == edit.target


def main() -> None:
    edit_list = get_edits(DELETED_ERRORS + SPELLING_ERRORS)

    models = run.get_all_models(finetuned_only=True)

    for model_name, (cls_model, model, tokenizer) in models.items():
        print(f"Working on {model_name.capitalize()}")
        set_top_3_words(edit_list, model_name, cls_model, model, tokenizer)

    spelling_edits = [e for e in edit_list if e.sent_id in SPELLING_ERRORS]
    deleted_edits = [e for e in edit_list if e.sent_id in DELETED_ERRORS]

    with open(disk.QUALITATIVE_MD_FILE, "w") as file:
        file.write(
            f"<!--\npandoc --out qualititative-study.pdf {disk.QUALITATIVE_MD_FILE} && open qualititative-study.pdf\n-->\n\n"
        )

        file.write("# Spelling Errors\n\n")

        file.writelines([to_markdown_row(edit) for edit in spelling_edits])

        file.write("# Deleted Errors\n\n")

        file.writelines([to_markdown_row(edit) for edit in deleted_edits])


if __name__ == "__main__":
    main()
