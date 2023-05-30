from torch import nn
from transformers import BertModel, DistilBertModel, AutoModel
from abc import abstractmethod


class BertClassifier(nn.Module):
    def __init__(self, model_name: str, dropout_prob: float):
        super(BertClassifier, self).__init__()
        self.module_name = model_name
        self.dropout = nn.Dropout(p=dropout_prob)

    @abstractmethod
    def forward(self, input_ids, attention_mask):
        pass


class CustomBertClassifier(BertClassifier):
    # bert core + dropout + one layer feed-forward
    def __init__(self, model_name, dropout_prob, n_classes):
        super(CustomBertClassifier, self).__init__(
            model_name=model_name, dropout_prob=dropout_prob
        )
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.classifier(output)


class CustomDistilBertClassifier(BertClassifier):
    def __init__(self, model_name, dropout_prob, n_classes):
        super(CustomDistilBertClassifier, self).__init__(
            model_name=model_name, dropout_prob=dropout_prob
        )
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_output = pooled_output[:, 0]
        output = self.dropout(pooled_output)
        return self.classifier(output)


class CustomTinyBertClassifier(BertClassifier):
    def __init__(self, model_name, dropout_prob, n_classes):
        super(CustomTinyBertClassifier, self).__init__(
            model_name=model_name, dropout_prob=dropout_prob
        )
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_output = pooled_output[:, 0]
        output = self.dropout(pooled_output)
        return self.classifier(output)
