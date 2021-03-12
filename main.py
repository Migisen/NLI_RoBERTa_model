from data.data_loader import SNLIDataModule
from model.bert_classifier import ClassifierRoBERT
from transformers import RobertaModel

import torch

# Для вычислений на GPU
device = torch.device(torch.cuda.current_device())
# Параметры модели


if __name__ == '__main__':
    model = ClassifierRoBERT(num_classes=3, drop_p=0.3).to(device)
    print('Done!')
