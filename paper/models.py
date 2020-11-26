import os
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import AdamW, AutoModelForSequenceClassification, BertPreTrainedModel

from . import tokenizers
from .structures import HyperParameters


class SentenceClassificationModel(nn.Module):
    def __init__(self, bert_model: BertPreTrainedModel) -> None:
        super().__init__()  # type: ignore
        self.bert = bert_model

    def forward(self, input_ids: Tensor, attn_mask: Tensor, **kwargs: Any) -> Tensor:
        if "output_attentions" not in kwargs:
            kwargs["output_attentions"] = False

        logits: Tensor = self.bert(input_ids, attention_mask=attn_mask, **kwargs)[0]
        return logits

    def get_optimizer(self, config: HyperParameters) -> Optimizer:
        return AdamW(self.parameters(), lr=config.learning_rate, eps=1e-8)  # type: ignore


def get_pretrained_model(config: HyperParameters) -> SentenceClassificationModel:
    """
    based on config.name, returns a vanilla pretrained model or an arxiv-specific model.
    """

    if "arxiv" in config.model_name:  # nice pattern matching
        possible_dir = config.models_dir / config.model_name

        if os.path.isdir(possible_dir):
            bert = AutoModelForSequenceClassification.from_pretrained(
                config.root_model_name
            )

        else:
            raise ValueError("Need pretrained model on disk.")
    else:
        bert = AutoModelForSequenceClassification.from_pretrained(config.model_name)

        bert.resize_token_embeddings(len(tokenizers.get_tokenizer(config)))

    return SentenceClassificationModel(bert)


def pretrain_prep(config: HyperParameters) -> None:
    # this code needs to create the model in the directory.
    assert "arxiv" in config.model_name

    tokenizer = tokenizers.get_tokenizer(config)

    bert = AutoModelForSequenceClassification.from_pretrained(config.root_model_name)

    bert.resize_token_embeddings(len(tokenizer))

    possible_dir = config.models_dir / config.model_name

    bert.save_pretrained(str(possible_dir))
    tokenizer.save_pretrained(str(possible_dir))

    print("You probably want to run language_model.sh now.")


if __name__ == "__main__":
    config = HyperParameters("./experiments/debugging/params.toml")
    for i in range(10):
        model = get_pretrained_model(config)
        print(model.bert.classifier.bias)
