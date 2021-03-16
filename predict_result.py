from model.bert_classifier import ClassifierRoBERT
from torch.utils.data import DataLoader
from data.snli_dataset import SNLIData
from transformers.utils import logging
from tqdm import tqdm as tq

import pandas as pd
import argparse
import torch
import os

# Disable transformers warnings
logging.set_verbosity_error()

# CLI
parser = argparse.ArgumentParser(description='Use this file to make predictions and produce result.csv.'
                                             'Place сheckpoint in \'chekpoints\' folder')

parser.add_argument('--batch_size', type=int, help='test batch size, choose wisely', default=8)
parser.add_argument('--checkpoint_name', help='filename of saved model', default='final_weights.ckpt')

args = parser.parse_args()

# Config
CHECKPOINT_NAME = args.checkpoint_name
BATCH_SIZE = args.batch_size


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import data
snli_test_data = SNLIData(data_name='snli_1.0_test')
snli_test_loader = DataLoader(dataset=snli_test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)

checkpoint_name = os.path.join('checkpoints', CHECKPOINT_NAME)
roberta_model = ClassifierRoBERT(num_classes=3).to(device)
roberta_model.load_state_dict(state_dict=torch.load(checkpoint_name))

results = []
pair_ids = []
true_labels = []
for i, batch in enumerate(tq(snli_test_loader)):
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
accuracy = (result.predicted_label == true_labels).sum()/len(true_labels)
print(f'Accuracy on test dataset = {accuracy}')
print('Saving results.csv...')
result.to_csv('./results.csv', index=False)
print('Done!')


