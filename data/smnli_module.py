from torch.utils.data import DataLoader

from data.mnli_module import MNLIDataModule
from data.smnli_dataset import SMNLIData


class SMNLIDataModule(MNLIDataModule):

    def setup(self, stage=None):
        self.train_dataset = SMNLIData(file_directory=self.file_directory,
                                       file_name=self.train_filename)
        self.val_dataset_1 = SMNLIData(file_directory=self.file_directory,
                                       file_name=self.val_filename)
        self.val_dataset_2 = SMNLIData(file_directory=self.file_directory,
                                       file_name=self.val_filename_2)

    # def train_dataloader(self):
    #     return DataLoader(dataset=self.train_dataset,
    #                       shuffle=True,
    #                       batch_size=self.batch_size,
    #                       num_workers=12)
    #
    # def val_dataloader(self):
    #     super(SMNLIDataModule, self).val_dataloader()
