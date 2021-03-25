from data.nli_dataset import NLIData
from torch.utils.data import DataLoader

import pytorch_lightning as pl


# noinspection PyAttributeOutsideInit
class SNLIDataModule(pl.LightningDataModule):

    def __init__(self, file_directory: str, train_filename: str, val_filename: str, test_filename=None, batch_size=8):
        super().__init__()
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.val_filename = val_filename
        self.batch_size = batch_size
        self.file_directory = file_directory

    def setup(self, stage=None):
        self.train_dataset = NLIData(file_directory=self.file_directory, file_name=self.train_filename)
        self.val_dataset = NLIData(file_directory=self.file_directory, file_name=self.val_filename)
        if self.test_filename is not None:
            self.test_dataset = NLIData(file_directory=self.file_directory, file_name=self.test_filename)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def prepare_data(self, *args, **kwargs):
        pass
