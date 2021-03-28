from model.bert_classifier import ClassifierRoBERT
from torch.utils.data import DataLoader
from data.nli_dataset import NLIData
from transformers.utils import logging
from tqdm import tqdm as tq

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем данные

nli_data = NLIData(file_directory=DATASET_DIR, file_name=DATASET_NAME)
nli_loader = DataLoader(dataset=nli_data, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

checkpoint_name = os.path.join('checkpoints', CHECKPOINT_NAME)
roberta_model = ClassifierRoBERT(num_classes=3).to(device)
roberta_model.load_state_dict(state_dict=torch.load(checkpoint_name))

results = []
pair_ids = []
true_labels = []
for i, batch in enumerate(tq(nli_loader)):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    label = batch['label'].flatten().to(device)
    true_labels += label.tolist()
    pair_ids += batch['pair_id']
    with torch.no_grad():
        prediction = roberta_model(input_ids, attention_mask, label).logits
        prediction.to('cpu')
        _, y_hat = torch.max(prediction, dim=1)
        results += y_hat.tolist()

result = pd.DataFrame({'pairID': pair_ids, 'predicted_label': results})
accuracy = (result.predicted_label == true_labels).sum() / len(true_labels)
print(f'Accuracy on test dataset = {accuracy}')
print(f'Saving {OUTPUT_NAME}.csv...')
result.to_csv(f'./{OUTPUT_NAME}.csv', index=False)
print('Done!')

if __name__ == '__main__':
    pass
