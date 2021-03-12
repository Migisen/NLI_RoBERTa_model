from data.snli_dataset import SNLIData
from torch.utils.data import DataLoader

import pytorch_lightning as pl


# noinspection PyAttributeOutsideInit
class SNLIDataModule(pl.LightningDataModule):

    def __init__(self, train_filename, test_filename, val_name, tokenizer=None, batch_size=8):
        super().__init__()
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.val_filename = val_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SNLIData(data_name=self.train_filename)
        self.test_dataset = SNLIData(data_name=self.test_filename)
        self.val_dataset = SNLIData(data_name=self.val_filename)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)
