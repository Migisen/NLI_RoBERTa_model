from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW

import pytorch_lightning as pl
import torch


class ClassifierRoBERT(pl.LightningModule):

    def __init__(self, num_classes, n_epoch=5, steps_per_epoch=None):
        super().__init__()
        self.n_epochs = n_epoch
        self.steps_per_epoch = steps_per_epoch
        self.config = RobertaConfig.from_pretrained('roberta-base', num_labels=num_classes)
        self.robert = RobertaForSequenceClassification.from_pretrained('roberta-base', config=self.config)

    def forward(self, input_ids, attention_mask, label):
        x = self.robert(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = 0
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

    # Обучение

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label, _ = batch.values()
        label = label.flatten()
        loss, prediction = self.forward(input_ids, attention_mask, label).values()
        accuracy = self.calculate_accuracy(prediction, label)
        self.log('train_accuracy', accuracy, prog_bar=False, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss, 'train_accuracy': accuracy}

    def training_epoch_end(self, outputs):
        mean_accuracy = self.calculate_mean_statistic(outputs, 'train_accuracy')
        self.log('val_mean_accuracy', mean_accuracy, prog_bar=False, logger=True)

    # Валидация

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, label, _ = batch.values()
        label = label.flatten()
        loss, prediction = self.forward(input_ids, attention_mask, label).values()
        accuracy = self.calculate_accuracy(prediction, label)
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

    # Тестирование

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch.values()
        label = label.flatten()
        loss, prediction = self.forward(input_ids, attention_mask, label).values()
        accuracy = self.calculate_accuracy(prediction, label)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return {'loss': loss, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        mean_accuracy = self.calculate_mean_statistic(outputs, 'test_accuracy')

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
