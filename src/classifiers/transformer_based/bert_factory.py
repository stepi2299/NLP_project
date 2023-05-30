from transformers import (
    PreTrainedTokenizerBase,
    BertTokenizer,
    DistilBertTokenizer,
    AutoTokenizer,
)
from src.classifiers.transformer_based.bert_models import (
    CustomBertClassifier,
    CustomTinyBertClassifier,
    CustomDistilBertClassifier,
    BertClassifier,
)
from typing import Tuple


class BertFactory:
    POSSIBLE_MODEL_NAMES = ["bert", "distil_bert", "tiny_bert"]

    @staticmethod
    def create(
        model_name: str, dropout_prob: float, n_classes: int
    ) -> Tuple[PreTrainedTokenizerBase, BertClassifier]:
        if not model_name.lower() in BertFactory.POSSIBLE_MODEL_NAMES:
            raise Exception(
                f"Received model is not supported, received: {model_name},"
                f" supported: {BertFactory.POSSIBLE_MODEL_NAMES}"
            )
        if model_name == "bert":
            whole_model_name = "bert-base-uncased"
            return (
                BertTokenizer.from_pretrained(whole_model_name),
                CustomBertClassifier(whole_model_name, dropout_prob, n_classes),
            )
        elif model_name == "distil_bert":
            whole_model_name = "distilbert-base-uncased"
            return (
                DistilBertTokenizer.from_pretrained(whole_model_name),
                CustomDistilBertClassifier(whole_model_name, dropout_prob, n_classes),
            )
        else:
            whole_model_name = "huawei-noah/TinyBERT_General_4L_312D"
            return (
                AutoTokenizer.from_pretrained(whole_model_name),
                CustomTinyBertClassifier(whole_model_name, dropout_prob, n_classes),
            )
