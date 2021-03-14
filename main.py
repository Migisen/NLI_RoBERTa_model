from data.data_loader import SNLIDataModule
from model.bert_classifier import ClassifierRoBERT

import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning as pl


# Параметры модели
NUM_CLASSES = 3
DATA_LEN = 549367
N_EPOCHS = 5
BATCH_SIZE = 16

if __name__ == '__main__':
    snli_data = SNLIDataModule('snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=BATCH_SIZE)
    robert_model = ClassifierRoBERT(
        num_classes=NUM_CLASSES,
        n_epoch=N_EPOCHS,
        steps_per_epoch=DATA_LEN // BATCH_SIZE
    )
    # Настройки логов
    wandb_logger = pl_loggers.WandbLogger(name='migi-pc-roBERTa',
                                          save_dir='logs',
                                          project='snli_roBERTa',
                                          entity='migisen')

    # Настройки чекпоинтов
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints',
                                                       filename='roBERTa_classifier',
                                                       save_last=True,
                                                       monitor='val_accuracy',
                                                       verbose=True,
                                                       mode='max')
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, callbacks=checkpoint_callback)
    trainer.fit(robert_model, snli_data)
    print('Done!')
