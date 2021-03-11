from data import data_loader
test = data_loader.SNLIData(data_name='snli_1.0_train')


if __name__ == '__main__':
    print(test.__getitem__(10))
    print('Done!')


