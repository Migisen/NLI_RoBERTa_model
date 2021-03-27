from data.nli_dataset import NLIData


class SMNLIData(NLIData):
    """
    Датасет MNLI для использования с sentence-roBERTa

    """

    def __init__(self, file_directory: str, file_name: str, max_length: int = 42):
        super().__init__(file_directory=file_directory, file_name=file_name, max_length=max_length)

    def __getitem__(self, idx) -> dict:
        """
        Токенизируем 2 предложения по отдельности (отличия sBERT)
        :param idx:
        :return:
        """
        tokenized_sentences = [self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length,
                                              padding='max_length', truncation=True) for sentence in self.data[idx]]
        judgment = self.data_labels[idx]
        pair_id = self.pair_id[idx]
        result = {'input_ids_0': tokenized_sentences[0]['input_ids'].flatten(),
                  'attention_mask_0': tokenized_sentences[0]['attention_mask'].flatten(),
                  'input_ids_1': tokenized_sentences[1]['input_ids'].flatten(),
                  'attention_mask_1': tokenized_sentences[1]['attention_mask'].flatten(),
                  'label': judgment,
                  'pair_id': pair_id,
                  }
        return result
