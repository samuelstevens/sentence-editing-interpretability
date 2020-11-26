import csv
from typing import List, NamedTuple

import torch
from torch import Tensor

from .structures import ConfusionMatrix

ROBERTA_FILE = "/Users/samstevens/Documents/school/subjects/research/thesis/data-versioned/roberta-base-aesw-32-1e-6-503176a5-8a15-4d3b-afa0-5e093934a611-test-inference.csv"

SCIBERT_FILE = "/Users/samstevens/Documents/school/subjects/research/thesis/data-versioned/scibert-1e-6-8128a162-5efb-4920-b63f-87a2be6fd503-test-inference.csv"

BERT_FILE = "/Users/samstevens/Documents/school/subjects/research/thesis/data-versioned/bert-base-aesw-32-1e-6-8ac1b935-200a-41fc-94a4-5d185aab1269-test-inference.csv"


class Answer(NamedTuple):
    bert: bool
    roberta: bool
    scibert: bool


class Example(NamedTuple):
    """
    Represents a single example.
    """

    identifier: str
    bert_sent: str
    roberta_sent: str
    scibert_sent: str
    label: bool
    bert: bool
    roberta: bool
    scibert: bool

    @property
    def answers(self) -> Answer:
        return Answer(
            self.bert == self.label,
            self.roberta == self.label,
            self.scibert == self.label,
        )


def get_examples() -> List[Example]:
    with open(BERT_FILE) as csvfile:
        reader = csv.reader(csvfile)
        bert = [
            (i, sent, bool(int(label)), bool(int(prediction)))
            for i, sent, label, prediction in reader
        ]

    with open(ROBERTA_FILE) as csvfile:
        reader = csv.reader(csvfile)
        roberta = [
            (i, sent, bool(int(label)), bool(int(prediction)))
            for i, sent, label, prediction in reader
        ]

    with open(SCIBERT_FILE) as csvfile:
        reader = csv.reader(csvfile)
        scibert = [
            (i, sent, bool(int(label)), bool(int(prediction)))
            for i, sent, label, prediction in reader
        ]

    assert len(scibert) == len(roberta) == len(bert) == 142923, f"{len(roberta)}"

    for i in range(len(scibert)):
        assert (
            scibert[i][2] == roberta[i][2] == bert[i][2]
        ), f"'{scibert[i][2]}' '{roberta[i][2]}' '{bert[i][2]}'"

    examples = []
    for i in range(len(roberta)):
        ident, _, label, _ = roberta[i]
        example = Example(
            ident,
            bert[i][1],
            roberta[i][1],
            scibert[i][1],
            label,
            bert[i][3],
            roberta[i][3],
            scibert[i][3],
        )

        examples.append(example)

    return examples


def report_results() -> None:
    examples = get_examples()

    roberta_TP = 0
    roberta_FP = 0
    roberta_FN = 0
    roberta_TN = 0

    scibert_TP = 0
    scibert_FP = 0
    scibert_FN = 0
    scibert_TN = 0

    bert_TP = 0
    bert_FP = 0
    bert_FN = 0
    bert_TN = 0

    for example in examples:
        if example.label and example.roberta:
            roberta_TP += 1
        if example.label and example.scibert:
            scibert_TP += 1
        if example.label and example.bert:
            bert_TP += 1

        if example.label and not example.roberta:
            roberta_FN += 1
        if example.label and not example.scibert:
            scibert_FN += 1
        if example.label and not example.bert:
            bert_FN += 1

        if not example.label and example.roberta:
            roberta_FP += 1
        if not example.label and example.scibert:
            scibert_FP += 1
        if not example.label and example.bert:
            bert_FP += 1

        if not example.label and not example.roberta:
            roberta_TN += 1
        if not example.label and not example.scibert:
            scibert_TN += 1
        if not example.label and not example.bert:
            bert_TN += 1

    roberta_confusion = ConfusionMatrix(roberta_TP, roberta_TN, roberta_FP, roberta_FN)
    assert roberta_confusion.total == 142923

    print("Roberta results:")
    print(roberta_confusion)
    print(f"Accuracy: {roberta_confusion.accuracy:.3f}")
    print(f"Precision: {roberta_confusion.precision:.3f}")
    print(f"Recall: {roberta_confusion.recall:.3f}")
    print(f"F1: {roberta_confusion.f1:.3f}")
    print()

    scibert_confusion = ConfusionMatrix(scibert_TP, scibert_TN, scibert_FP, scibert_FN)
    assert scibert_confusion.total == 142923

    print("Scibert results:")
    print(scibert_confusion)
    print(f"Accuracy: {scibert_confusion.accuracy:.3f}")
    print(f"Precision: {scibert_confusion.precision:.3f}")
    print(f"Recall: {scibert_confusion.recall:.3f}")
    print(f"F1: {scibert_confusion.f1:.3f}")
    print()

    bert_confusion = ConfusionMatrix(bert_TP, bert_TN, bert_FP, bert_FN)
    assert bert_confusion.total == 142923

    print("Bert results:")
    print(bert_confusion)
    print(f"Accuracy: {bert_confusion.accuracy:.3f}")
    print(f"Precision: {bert_confusion.precision:.3f}")
    print(f"Recall: {bert_confusion.recall:.3f}")
    print(f"F1: {bert_confusion.f1:.3f}")
    print()

    print("Positive:", len([ex for ex in examples if ex.label]))
    print("Negative:", len([ex for ex in examples if not ex.label]))


def get_confusion_matrix(logits: Tensor, labels: Tensor) -> ConfusionMatrix:
    _, predicted = torch.max(logits.data, 1)  # type: ignore

    true_pos: int = ((predicted * labels) != 0).sum().item()
    true_neg: int = (((predicted - 1) * (labels - 1)) != 0).sum().item()
    false_pos: int = ((predicted * (labels - 1)) != 0).sum().item()
    false_neg: int = (((predicted - 1) * labels) != 0).sum().item()

    confusion = ConfusionMatrix(true_pos, true_neg, false_pos, false_neg)

    assert confusion.total == len(
        predicted
    ), f"confusion matrix {confusion} doesn't have the same number of results as {labels}: {confusion.total} != {len(labels)}"

    return confusion


def main() -> None:
    report_results()


if __name__ == "__main__":
    main()
