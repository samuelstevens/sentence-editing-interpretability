import json
from dataclasses import dataclass
from typing import Any, Optional

from . import predict


@dataclass
class Score:
    model_name: str
    strategy: predict.Strategy
    exact_match_accuracy: float
    jaccard_match_accuracy: float
    avg_jaccard_similarity: float
    top_3_match_accuracy: float
    skipped_false_predictions: bool

    @staticmethod
    def parse(dct: Any) -> Optional["Score"]:
        if "__Score__" in dct and dct["__Score__"]:
            # parse it
            return Score(
                dct["model name"],
                dct["strategy"],
                dct["exact match accuracy"],
                dct["jaccard match accuracy"],
                dct["avg jaccard similarity"],
                dct["top 3 match accuracy"],
                dct["skipped false predictions"],
            )
        else:
            return None


class ScoreEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if not isinstance(o, Score):
            return super().default(o)

        return {
            "__Score__": True,
            "model name": o.model_name,
            "strategy": o.strategy,
            "exact match accuracy": o.exact_match_accuracy,
            "jaccard match accuracy": o.jaccard_match_accuracy,
            "avg jaccard similarity": o.avg_jaccard_similarity,
            "top 3 match accuracy": o.top_3_match_accuracy,
            "skipped false predictions": o.skipped_false_predictions,
        }
