from torch.utils.data import Dataset
from torch import LongTensor
from transformers import RobertaTokenizer

import json
import os


class NLIData(Dataset):
    """
    NLI dataset converter
    Attributes:
        file_name (str): dataset file name (e.g. 'snli_1.0_train')
        tokenizer (transformers.PreTrainedTokenizer, optional): tokenizer from transformers library
        max_length (int, optional): length to which sentences are truncated
    """

    def __init__(self, file_directory: str, file_name: str, max_length=42):
        """
        Загрузка датасета из jsonl
        Args:
            file_directory (str): Путь к файлу датасета.
            file_name (str): Название датасета.
            max_length (int, optional): Максимальный размер предложения.
        """
        self.data = []
        self.data_labels = []
        self.pair_id = []
        self.label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.data_path = os.path.join(os.path.dirname(__file__), file_directory, f'{file_name}.jsonl')
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['gold_label'] not in self.label_map:
                    continue
                self.pair_id.append(line['pairID'])
                self.data.append((line['sentence1'], line['sentence2']))
                self.data_labels.append(LongTensor([self.label_map[line['gold_label']]]))
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data_labels)

    def __getitem__(self, idx) -> dict:
        text, hypothesis = self.data[idx]
        judgment = self.data_labels[idx]
        result_item = self.tokenizer(text, hypothesis, return_tensors='pt', max_length=self.max_length,
                                     padding='max_length', truncation=True)  # 99% квантиль: 175 знака / 42 слова
        pair_id = self.pair_id[idx]
        return {'input_ids': result_item['input_ids'].flatten(),
                'attention_mask': result_item['attention_mask'].flatten(),
                'label': judgment,
                'pair_id': pair_id}

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
