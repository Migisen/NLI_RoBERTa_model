import torch
from transformers import RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

import pytorch_lightning as pl
import torch.nn as nn


class ClassifierRoBERT(pl.LightningModule):

    def __init__(self, num_classes, drop_p, n_epoch, steps_per_epoch=None):
        super().__init__()
        self.n_epochs = n_epoch
        self.steps_per_epoch = steps_per_epoch
        self.robert = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=drop_p)
        self.out = nn.Linear(self.robert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        pooled_output = self.robert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch.values()
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)  # Берем класс с самой большой вероятностью
        loss = self.criterion(output, label.flatten())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch.values()
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        loss = self.criterion(output, label.flatten())
        self.log('val_loss', loss, prog_bar=False, logger=True)
        return loss
