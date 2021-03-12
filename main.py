from data.data_loader import SNLIDataModule
from model.bert_classifier import ClassifierRoBERT

import pytorch_lightning as pl

# Параметры модели
NUM_CLASSES = 3
DATA_LEN = 549367
N_EPOCHS = 5
BATCH_SIZE = 16
DROP_RATE = 0.2

if __name__ == '__main__':
    snli_data = SNLIDataModule('snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=BATCH_SIZE)
    robert_model = ClassifierRoBERT(
        num_classes=NUM_CLASSES,
        drop_p=DROP_RATE,
        n_epoch=N_EPOCHS,
        steps_per_epoch=DATA_LEN // BATCH_SIZE
    )
    trainer = pl.Trainer(gpus=1)
    trainer.fit(robert_model, snli_data)
    print('Done!')
