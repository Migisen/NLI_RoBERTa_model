from torch.utils.data import Dataset
from torch import LongTensor
from transformers import RobertaTokenizer

import json
import os


class SNLIData(Dataset):
    """
    SNLI dataset converter
    Attributes:
        data_name (str): dataset file name (e.g. 'snli_1.0_train')
        tokenizer (transformers.PreTrainedTokenizer, optional): tokenizer from transformers library
        max_length (int, optional): length to which sentences are truncated
    """

    def __init__(self, data_name, tokenizer=None, max_length=42):
        self.data = []
        self.data_labels = []
        self.label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.data_path = os.path.join(os.path.dirname(__file__), 'snli', f'{data_name}.jsonl')
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['gold_label'] not in self.label_map:
                    continue
                self.data.append((line['sentence1'], line['sentence2']))
                self.data_labels.append(LongTensor([self.label_map[line['gold_label']]]))

        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        text, hypothesis = self.data[idx]
        judgment = self.data_labels[idx]
        result_item = self.tokenizer(text, hypothesis, return_tensors='pt', max_length=self.max_length,
                                     padding='max_length', truncation=True)  # 99% квантиль: 175 знака / 42 слова
        return {'input_ids': result_item['input_ids'].flatten(),
                'attention_mask': result_item['attention_mask'].flatten(),
                'label': judgment}

    @staticmethod
    def pad_input(text: str, hypothesis: str) -> str:
        """
        Any preprocessing can be done here
        :param text: first sentence
        :param hypothesis: second sentence
        :return:
        """
        result: str = f'{text} {hypothesis}'
        return result
