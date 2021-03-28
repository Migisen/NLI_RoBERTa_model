from data.mnli_module import MNLIDataModule
from data.nli_dataset import NLIData
from model.bert_classifier import ClassifierRoBERT
from transformers import logging

import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning as pl

# Отключаем предупреждения о весах
logging.set_verbosity_error()


def train_on_mnli(batch_size, n_classes, n_epochs, gpu):
    # Считаем длину для рассчета числа шагов в эпохе
    data_len = len(NLIData(file_directory='mnli', file_name='multinli_1.0_train'))
    steps_per_epoch = round(data_len / batch_size)

    # Подготавливаем данные
    mnli_data = MNLIDataModule(file_directory='mnli', train_filename='multinli_1.0_train',
                               val_matched_filename='multinli_1.0_dev_matched',
                               val_mismatched_filename='multinli_1.0_dev_mismatched',
                               batch_size=batch_size)

    # Инициализируем модель
    robert_model = ClassifierRoBERT(
        num_classes=n_classes,
        n_epoch=n_epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Настройки логов
    wandb_logger = pl_loggers.WandbLogger(save_dir='../logs',
                                          project='mnli_roBERTa',
                                          entity='migisen')

    # Настройки чекпоинтов
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints',
                                                       filename='roBERTa_mnli_classifier',
                                                       save_last=True,
                                                       monitor='val_matched_mean_accuracy',
                                                       verbose=True,
                                                       mode='max')

    trainer = pl.Trainer(gpus=gpu, logger=wandb_logger, callbacks=checkpoint_callback)
    trainer.fit(robert_model, mnli_data)
    print('Done!')


if __name__ == '__main__':
    pass
