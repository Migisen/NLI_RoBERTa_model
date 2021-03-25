from data.snli_module import SNLIDataModule
from torch.utils.data import DataLoader
from data.nli_dataset import NLIData


class MNLIDataModule(SNLIDataModule):
    def __init__(self, file_directory: str, train_filename: str, val_matched_filename: str,
                 val_mismatched_filename: str,
                 batch_size: int = 8):
        super().__init__(file_directory=file_directory, train_filename=train_filename,
                         val_filename=val_matched_filename,
                         batch_size=batch_size)
        self.val_filename_2 = val_mismatched_filename

    def setup(self, stage=None):
        super().setup()
        self.val_dataset_2 = NLIData(file_directory=self.file_directory, file_name=self.val_filename_2)

    def val_dataloader(self):
        val_loader_1 = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
        val_loader_2 = DataLoader(self.val_dataset_2, batch_size=self.batch_size, num_workers=2)
        return [val_loader_1, val_loader_2]
