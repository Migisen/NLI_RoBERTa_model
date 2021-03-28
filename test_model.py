from model.bert_classifier import ClassifierRoBERT
from data.snli_module import SNLIDataModule
from transformers import logging

import pytorch_lightning as pl
import argparse
import torch
import os

# Disable transformers warnings
logging.set_verbosity_error()

# CLI
parser = argparse.ArgumentParser(description='Evaluate model on chosen dataset. '
                                             'Place weights in \'checkpoints\' folder.''')
parser.add_argument('batch_size', type=int, help='test batch size, choose wisely')
parser.add_argument('--dataset_name', choices=['train', 'test', 'dev'],
                    help='select dataset to test', default='test')
parser.add_argument('--checkpoint_name', help='filename of saved model', default='final_weights.ckpt')
parser.add_argument('--gpus', help='select number of gpus', default=1)
args = parser.parse_args()

# Import data

snli_data = SNLIDataModule('snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=args.batch_size)
snli_data.setup()

if args.dataset_name == 'test':
    dataloader = snli_data.test_dataloader()
elif args.dataset_name == 'train':
    dataloader = snli_data.train_dataloader()
else:
    dataloader = snli_data.val_dataloader()

checkpoint_path = os.path.join('checkpoints', args.checkpoint_name)

roberta_model = ClassifierRoBERT(num_classes=3)
roberta_model.load_state_dict(state_dict=torch.load(checkpoint_path))
trainer = pl.Trainer(gpus=args.gpus)
test_results = trainer.test(roberta_model, test_dataloaders=dataloader)[0]['test_accuracy']
print(f'Model accuracy on {args.dataset_name} is {test_results}')
