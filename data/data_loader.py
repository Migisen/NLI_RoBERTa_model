from torch.utils.data import Dataset
from transformers import RobertaTokenizer

import json
import os


class SNLIData(Dataset):
    def __init__(self, data_name, tokenizer=None):
        self.data = []
        self.data_labels = []
        self.label_map = {'contradiction': -1, 'neutral': 0, 'entailment': 1}
        self.data_path = os.path.join(os.path.dirname(__file__), 'snli', f'{data_name}.jsonl')
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['gold_label'] not in self.label_map:
                    continue
                self.data.append((line['sentence1'], line['sentence2']))
                self.data_labels.append(line['gold_label'])
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        text, hypothesis = self.data[idx]
        judgment = self.data_labels[idx]
        return self.tokenizer(text, return_tensors='pt')
