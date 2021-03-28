from data.snli_module import SNLIDataModule
from torch.utils.data import DataLoader
from data.nli_dataset import NLIData


class MNLIDataModule(SNLIDataModule):
    """Lightning модуль для формирования загрузчика данных MNLI датасета

    """

    def __init__(self, file_directory: str, train_filename: str, val_matched_filename: str,
                 val_mismatched_filename: str,
                 batch_size: int = 8):
        """

        :param file_directory: (str): путь к папке датасета.
        :param train_filename: (str): название файла обучающего датасета (без расширения).
        :param val_matched_filename: (str): название файла matched валидационного датасета (без расширения).
        :param val_mismatched_filename: (str): название файла mismatched валидационного датасета (без расширения).
        :param batch_size: (optional, int): размер батча.
        """
        super().__init__(file_directory=file_directory, train_filename=train_filename,
                         val_filename=val_matched_filename,
                         batch_size=batch_size)
        self.val_filename_2 = val_mismatched_filename

    def setup(self, stage=None):
        super().setup()
        self.val_dataset_2 = NLIData(file_directory=self.file_directory, file_name=self.val_filename_2)

    def val_dataloader(self):
        val_loader_1 = DataLoader(self.val_dataset_1, batch_size=self.batch_size, num_workers=12)
        val_loader_2 = DataLoader(self.val_dataset_2, batch_size=self.batch_size, num_workers=12)
        return [val_loader_1, val_loader_2]
