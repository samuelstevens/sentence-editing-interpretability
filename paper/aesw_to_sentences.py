import argparse
import csv
import heapq
import os
from pathlib import Path
from typing import List, Tuple

from lxml import etree
from tqdm import tqdm


def extract_sentence(sent_elem: etree._Element) -> str:
    assert sent_elem.tag == "sentence"
    string_builder = [str(sent_elem.text) if sent_elem.text else ""]

    for del_ins in sent_elem:
        if del_ins.tag == "del" and del_ins.text:
            string_builder.append(str(del_ins.text))
        if del_ins.tail:
            string_builder.append(str(del_ins.tail))

    return "".join(string_builder)


def extract_sentence_after_editing(sent_elem: etree._Element) -> str:
    assert sent_elem.tag == "sentence"
    string_builder = [str(sent_elem.text) if sent_elem.text else ""]

    for del_ins in sent_elem:
        if del_ins.tag == "ins" and del_ins.text:
            string_builder.append(str(del_ins.text))
        if del_ins.tail:
            string_builder.append(str(del_ins.tail))

    return "".join(string_builder)


def extract_sentence_id(sent_elem: etree._Element) -> str:
    assert sent_elem.tag == "sentence"
    assert "sid" in sent_elem.attrib
    return str(sent_elem.attrib["sid"])


def get_label(sent_elem: etree._Element) -> int:
    """
    0 if no edit/no change, 1 if changed/edited
    """
    return 1 if len(list(sent_elem)) > 0 else 0


def read_xml(xml_filepath: Path, output_filepath: Path) -> None:
    if os.path.isfile(output_filepath):
        os.remove(output_filepath)

    with open(output_filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "sentence", "label"])

    sentences: List[Tuple[int, str, str, int]] = []

    for event, sent_elem in tqdm(etree.iterparse(str(xml_filepath), tag="sentence")):
        sent_id = extract_sentence_id(sent_elem)
        sent = extract_sentence(sent_elem)
        heapq.heappush(sentences, (len(sent), sent_id, sent, get_label(sent_elem)))

    with open(output_filepath, "a") as file:
        writer = csv.writer(file)
        while sentences:
            _, identifier, sent, label = heapq.heappop(sentences)
            writer.writerow((identifier, sent, label))


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datasets",
        help="which datasets (train, test, val) will be turned into .csv files.",
    )

    args = parser.parse_args()

    types = args.datasets.split(",")

    for t in types:
        if t not in ["train", "val", "test"]:
            print("--preprocess only accepts 'train', 'val' or 'test'.")

    if "test" in types:
        print("Converting test data...")
        test_xml = Path("./data-unversioned/aesw/aesw2016(v1.2)_test_modified.xml")
        test_out = Path("./data-unversioned/aesw/aesw-test.csv")
        read_xml(test_xml, test_out)
    if "val" in types:
        print("Converting validation data...")
        test_xml = Path("./data-unversioned/aesw/aesw2016(v1.2)_dev.xml")
        test_out = Path("./data-unversioned/aesw/aesw-dev.csv")
        read_xml(test_xml, test_out)
    if "train" in types:
        print("Converting training data...")
        test_xml = Path("./data-unversioned/aesw/aesw2016(v1.2)_train.xml")
        test_out = Path("./data-unversioned/aesw/aesw-train.csv")
        read_xml(test_xml, test_out)


if __name__ == "__main__":
    main()
