from data.mnli_module import MNLIDataModule
from data.nli_dataset import NLIData
from model.bert_classifier import ClassifierRoBERT
from transformers import logging

import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning as pl

# Disable transformers warnings
logging.set_verbosity_error()

# Params
BATCH_SIZE = 16
NUM_CLASSES = 3
N_EPOCHS = 5
DATA_LEN = len(NLIData(file_directory='mnli', file_name='multinli_1.0_train'))
GPUS = 1

mnli_data = MNLIDataModule(file_directory='mnli', train_filename='multinli_1.0_train',
                           val_matched_filename='multinli_1.0_dev_matched',
                           val_mismatched_filename='multinli_1.0_dev_mismatched',
                           batch_size=BATCH_SIZE)

robert_model = ClassifierRoBERT(
    num_classes=NUM_CLASSES,
    n_epoch=N_EPOCHS,
    steps_per_epoch=DATA_LEN // BATCH_SIZE
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
trainer = pl.Trainer(gpus=GPUS, logger=wandb_logger, callbacks=checkpoint_callback)
trainer.fit(robert_model, mnli_data)
print('Done!')
