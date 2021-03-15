from model.bert_classifier import ClassifierRoBERT

import torch
import os

checkpoint_path = os.path.join('../','checkpoints')
model = ClassifierRoBERT.load_from_checkpoint(os.path.join(checkpoint_path, 'roBERTa_classifier_1.ckpt'),
                                              num_classes=3, n_epoch=5)
torch.save(model.state_dict(), os.path.join(checkpoint_path, 'final_weights.ckpt'))