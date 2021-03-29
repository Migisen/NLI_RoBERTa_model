from data.snli_module import SNLIDataModule
from model.bert_classifier import ClassifierRoBERT
from transformers import logging

import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning as pl
import argparse

# todo: Переделать

# Disable transformers warnings
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description='Use this file to fine-tune model')
parser.add_argument('-n', '--num_classes', type=int, default=3, help='target number of classes')
parser.add_argument('-e', '--num_epochs', type=int, default=4, help='number of epochs to train')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='train batch size, choose wisely')
parser.add_argument('--wandb_entity', type=str, default='enter-name', help='wandb name')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

args = parser.parse_args()

# Параметры модели
NUM_CLASSES = args.num_classes
DATA_LEN = 549367
N_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
GPUS = args.gpus
WANDB_ENTITY = args.wandb_entity

if __name__ == '__main__':
    snli_data = SNLIDataModule('snli', 'snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=BATCH_SIZE)
    robert_model = ClassifierRoBERT(
        num_classes=NUM_CLASSES,
        n_epoch=N_EPOCHS,
        steps_per_epoch=DATA_LEN // BATCH_SIZE
    )
    # Настройки логов
    wandb_logger = pl_loggers.WandbLogger(save_dir='logs',
                                          project='snli_roBERTa',
                                          entity=WANDB_ENTITY,
                                          offline=True)

    # Настройки чекпоинтов
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints',
                                                       filename='roBERTa_classifier',
                                                       save_last=True,
                                                       monitor='val_accuracy',
                                                       verbose=True,
                                                       mode='max')
    trainer = pl.Trainer(gpus=GPUS, logger=wandb_logger, callbacks=checkpoint_callback)
    trainer.fit(robert_model, snli_data)
    print('Done!')
