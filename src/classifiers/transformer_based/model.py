from torch import nn
from transformers import BertModel


class CustomBertClassifier(nn.Module):
    # bert core + dropout + one layer feed-forward
    def __init__(self, model_name, dropout_prob, n_classes=2):
        super(CustomBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.classifier(output)
