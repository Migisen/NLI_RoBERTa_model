from model.bert_classifier import ClassifierRoBERT

import torch
import os

checkpoint_path = os.path.join('../', 'checkpoints')
model = ClassifierRoBERT.load_from_checkpoint(os.path.join(checkpoint_path, 'roBERTa_mnli_classifier.ckpt'),
                                              num_classes=3, n_epoch=5)
torch.save(model.state_dict(), os.path.join(checkpoint_path, 'mnli_weights.ckpt'))
