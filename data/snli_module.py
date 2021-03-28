from data.nli_dataset import NLIData
from torch.utils.data import DataLoader

import pytorch_lightning as pl


# noinspection PyAttributeOutsideInit
class SNLIDataModule(pl.LightningDataModule):
    """ Lightning модуль для формирование загрузчиков данных SNLI датасета

    """

    def __init__(self, file_directory: str, train_filename: str, val_filename: str, test_filename=None, batch_size=8):
        """

        :param file_directory: путь к папке датасета.
        :param train_filename: название файла обучающего датасета (без расширения).
        :param val_filename: название файла валидационного датасета (без расширения).
        :param test_filename: название файла тестового датасета (без расширения).
        :param batch_size: размер батча.
        """
        super().__init__()
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.val_filename = val_filename
        self.batch_size = batch_size
        self.file_directory = file_directory

    def setup(self, stage=None) -> None:
        """
        Инициализация датасетов (вызывается при обучении)
        """
        self.train_dataset = NLIData(file_directory=self.file_directory, file_name=self.train_filename)
        self.val_dataset_1 = NLIData(file_directory=self.file_directory, file_name=self.val_filename)
        if self.test_filename is not None:
            self.test_dataset = NLIData(file_directory=self.file_directory, file_name=self.test_filename)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=12)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset_1, batch_size=self.batch_size, num_workers=12)

    def prepare_data(self, *args, **kwargs):
        pass
