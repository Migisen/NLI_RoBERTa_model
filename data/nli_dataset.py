from torch.utils.data import Dataset
from torch import LongTensor
from transformers import RobertaTokenizer

import json
import os


class NLIData(Dataset):
    """
    Базовый класс NLI датасета

    """

    def __init__(self, file_directory: str, file_name: str, max_length=42, test_mode=False):
        """
        Загрузка датасета из jsonl
        :param  file_directory: (str): путь к файлу датасета.
        :param  file_name: (str): название датасета.
        :param  max_length: (int, optional): максимальный размер предложения.
        :param  test_mode: (bool, optional): загрузка тестового датасета без gold_label
        """
        self.data = []
        self.data_labels = []
        self.pair_id = []
        self.label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.data_path = os.path.join(os.path.dirname(__file__), file_directory, f'{file_name}.jsonl')
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['gold_label'] not in self.label_map and not test_mode:
                    continue
                self.pair_id.append(line['pairID'])
                self.data.append((line['sentence1'], line['sentence2']))
                if test_mode:
                    self.data_labels.append(LongTensor([0]))
                    continue
                self.data_labels.append(LongTensor([self.label_map[line['gold_label']]]))
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data_labels)

    def __getitem__(self, idx) -> dict:
        """

        :param idx: int: id объекта
        :return: dict: возвращает данные для загрузки в модель
        """
        text, hypothesis = self.data[idx]
        judgment = self.data_labels[idx]
        result_item = self.tokenizer(text, hypothesis, return_tensors='pt', max_length=self.max_length,
                                     padding='max_length', truncation=True)  # 99% квантиль: 175 знака / 42 слова
        pair_id = self.pair_id[idx]
        return {'input_ids': result_item['input_ids'].flatten(),
                'attention_mask': result_item['attention_mask'].flatten(),
                'label': judgment,
                'pair_id': pair_id}
