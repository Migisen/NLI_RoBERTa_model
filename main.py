from data import snli_dataset
from torch.utils.data import DataLoader

test_data = snli_dataset.SNLIData(data_name='snli_1.0_train')


if __name__ == '__main__':
    test_loader = DataLoader(test_data, batch_size=100)
    print(test_data.__getitem__(10))
    print('Done!')


