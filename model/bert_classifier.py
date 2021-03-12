from transformers import RobertaModel

import torch.nn as nn


class ClassifierRoBERT(nn.Module):

    def __init__(self, num_classes, drop_p, ):
        super().__init__()
        self.robert = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=drop_p)
        self.out = nn.Linear(self.robert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.robert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)
