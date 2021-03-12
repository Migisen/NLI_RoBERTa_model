from data.data_loader import SNLIDataModule
from transformers import RobertaModel

data_module = SNLIDataModule('snli_1.0_train', 'snli_1.0_test', 'snli_1.0_dev', batch_size=8)
data_module.setup()
train_sample = data_module.train_dataloader()
print(len(data_module.train_dataset))
for data in train_sample:
    input_test = data['input_ids']
    mask_test = data['attention_mask']
    roberta = RobertaModel.from_pretrained('roberta-base')
    roberta_pooled = roberta(input_ids=input_test, attention_mask=mask_test).pooler_output
    print('Размерность входных данных: {}'.format(input_test.shape))
    print('Размерность маски: {}'.format(mask_test.shape))
    print('Размерность таргета: {}'.format(data['label'].shape))
    print(f'Число скрытых узлов: {roberta_pooled.shape}')
    break
