from model.bert_classifier import ClassifierRoBERT
from torch.utils.data import DataLoader
from data.nli_dataset import NLIData
from transformers.utils import logging
from tqdm import tqdm as tq

import torch.nn as nn
import pandas as pd
import argparse
import torch
import os

"""
    Интерфейс для получение предсказаний из модели
    
    - Поддерживает и snli и mnli
    - Можно загружать разные веса из папки checkpoints
"""

# Disable transformers warnings
logging.set_verbosity_error()

# CLI
parser = argparse.ArgumentParser(description='Use this file to make predictions and produce result.csv.'
                                             'Place сheckpoint in \'chekpoints\' folder')
parser.add_argument('test_mode', type=bool, help='set to True if you want to predict on test (without labels) dataset')
parser.add_argument('-d', '--dataset_dir', type=str, help='dataset folder', default='snli', choices=('snli', 'mnli'))
parser.add_argument('-n', '--dataset_name', type=str, help='dataset filename', default='snli_1.0_test')
parser.add_argument('-b', '--batch_size', type=int, help='test batch size, choose wisely', default=8)
parser.add_argument('-c', '--checkpoint_name', help='filename of saved model', default='final_weights.ckpt')
parser.add_argument('-o', '--output_name', help='name of the output file', default='result')

args = parser.parse_args()

# Конфиг

CHECKPOINT_NAME = args.checkpoint_name
BATCH_SIZE = args.batch_size
DATASET_DIR = args.dataset_dir
DATASET_NAME = args.dataset_name
OUTPUT_NAME = args.output_name
TEST_MODE = args.test_mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем данные

nli_data = NLIData(file_directory=DATASET_DIR, file_name=DATASET_NAME, test_mode=TEST_MODE)
nli_loader = DataLoader(dataset=nli_data, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

checkpoint_name = os.path.join('checkpoints', CHECKPOINT_NAME)
roberta_model = ClassifierRoBERT(num_classes=3).to(device)
roberta_model.load_state_dict(state_dict=torch.load(checkpoint_name))
softmax = nn.Softmax(dim=1)

results = []
pair_ids = []
true_labels = []
confidence_list = []
for i, batch in enumerate(tq(nli_loader)):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    label = batch['label'].flatten().to(device)
    true_labels += label.tolist()
    pair_ids += batch['pair_id']
    with torch.no_grad():
        prediction = roberta_model(input_ids, attention_mask, label).logits
        prediction.to('cpu')
        probabilities = softmax(prediction)
        confidence, y_hat = torch.max(probabilities, dim=1)
        confidence_list.append(torch.mean(confidence))
        results += y_hat.tolist()

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
result = pd.DataFrame({'pairID': pair_ids, 'gold_label': results})
result.replace({'gold_label': label_map}, inplace=True)
print(f'Saving {OUTPUT_NAME}.csv...')
result.to_csv(f'./{OUTPUT_NAME}.csv', index=False)
print(sum(confidence_list) / len(confidence_list))
print('Done!')

if __name__ == '__main__':
    pass
