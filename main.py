from data import snli_dataset
from data.data_loader import SNLIDataModule
from torch.utils.data import DataLoader


#test_data = snli_dataset.SNLIData(data_name='snli_1.0_train')


if __name__ == '__main__':
    data_module = SNLIDataModule('snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=8)
    #print(test_data.__getitem__(10))
    print('Done!')


