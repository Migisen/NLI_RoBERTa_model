from transformers import RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW

import pytorch_lightning as pl
import torch.nn as nn
import torch


class SentenceRoBERTaClassifier(pl.LightningModule):
    def __init__(self, steps_per_epoch, num_classes: int = 3, n_epochs: int = 5):
        super().__init__()
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.roberta_config = RobertaConfig()
        # Инициализируем сиамскую сеть
        self.roberta = RobertaModel.from_pretrained('prajjwal1/roberta-base-mnli')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(3 * self.roberta_config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids_0, input_ids_1, attention_mask_0, attention_mask_1, label, **kwargs):
        u = torch.mean(self.roberta(input_ids_0, attention_mask_0).last_hidden_state, dim=1)
        v = torch.mean(self.roberta(input_ids_1, attention_mask_1).last_hidden_state, dim=1)
        output = torch.cat((u, v, torch.abs(u - v)), dim=1)
        output = self.linear(output)
        output = self.dropout(output)
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = 0
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        prediction = self.forward(**batch)
        label = batch['label'].flatten()
        accuracy = self.calculate_accuracy(prediction, label)
        loss = self.loss_fn(prediction, label)
        self.log('train_accuracy', accuracy, prog_bar=True)
        return {'loss': loss, 'train_accuracy': accuracy}

    def training_epoch_end(self, outputs):
        mean_accuracy = self.calculate_mean_statistic(outputs, 'train_accuracy')
        self.log('train_mean_accuracy', mean_accuracy, prog_bar=False, logger=True)

    # Валидация

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prediction = self.forward(**batch)
        label = batch['label'].flatten()
        accuracy = self.calculate_accuracy(prediction, label)
        loss = self.loss_fn(prediction, label)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return {'loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        if all(isinstance(elem, list) for elem in outputs):
            dataloader_names = {0: 'val_matched', 1: 'val_mismatched'}
            for dataloader_idx, dataset in enumerate(outputs):
                mean_accuracy = self.calculate_mean_statistic(dataset, 'val_accuracy')
                self.log(f'{dataloader_names[dataloader_idx]}_mean_accuracy', mean_accuracy, prog_bar=False,
                         logger=True)
        else:
            mean_accuracy = self.calculate_mean_statistic(outputs, 'val_accuracy')
            self.log('val_mean_accuracy', mean_accuracy, prog_bar=False, logger=True)

    @staticmethod
    def calculate_accuracy(prediction, label):
        _, y_hat = torch.max(prediction, dim=1)
        accuracy = (y_hat == label).sum() / len(label)
        return accuracy

    @staticmethod
    def calculate_mean_statistic(outputs, statistics_name: str):
        mean_statistics = 0
        for output in outputs:
            mean_statistics += output[statistics_name]
        mean_statistics /= len(outputs)
        return mean_statistics
