import json
from pathlib import Path
from typing import List

from .structures import Score

INTERPRETABILITY_FOLDER = Path("./data-unversioned/attention-weights/")
INTERPRETABILITY_FOLDER.mkdir(exist_ok=True)


SCORES_FOLDER = INTERPRETABILITY_FOLDER / "scores"
COMMA_SCORES_FILE = "comma-edit.json"
DELETE_SCORES_FILE = "delete-edit.json"
SPELLING_SCORES_FILE = "spelling-edit.json"

DELETE_EDIT_WEIGHTS_FILENAME = "deleted.pckl"
COMMA_EDIT_WEIGHTS_FILENAME = "comma.pckl"
SPELLING_EDIT_WEIGHTS_FILENAME = "spelling.pckl"

DEV_XML = "./data-unversioned/aesw/aesw2016(v1.2)_dev.xml"


QUALITATIVE_MD_FILE = INTERPRETABILITY_FOLDER / "qualititative-study.md"


def get_scores(filename: str) -> List[Score]:
    assert filename.endswith(".json")

    with open(SCORES_FOLDER / filename) as file:
        score_dict_list = json.load(file)

    parsed = [Score.parse(score) for score in score_dict_list]
    return [s for s in parsed if s]
