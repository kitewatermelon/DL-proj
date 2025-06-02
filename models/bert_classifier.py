import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output).squeeze(-1)
