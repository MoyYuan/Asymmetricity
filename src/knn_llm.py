from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample, util
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from enum import Enum
import numpy as np
from datasets import load_dataset

torch.manual_seed(0)

import json, time, random, traceback

from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict

mnli_labels = ['entailment', 'neutral', 'contradiction']

mnli_mapping = {0: 'entailment',
                1: 'neutral',
                2: 'contradiction'}

class KNNLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 0.5):
        super(KNNLoss, self).__init__()
        self.model = model
        self.margin = margin

    def get_config_dict(self):
        return 

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        h, p, r = reps
        
        distances = 1-F.cosine_similarity(torch.mul(h, r), p)

        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))

        return losses.mean()
    
def train():
    random.seed(0)

    model = SentenceTransformer('roberta-large')
    data = load_dataset("multi_nli")

    training_data = data['train']
    validation_data = data['validation_matched']

    train_examples = []

    for entry in training_data:
        h = entry['hypothesis']
        p = entry['premise']
        r = mnli_mapping[entry['label']]

        for mnli_label in mnli_labels:
            if r == mnli_label:
                train_examples.append(InputExample(texts=[h, p, r], label=1))
            else:
                train_examples.append(InputExample(texts=[h, p, r], label=0))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = KNNLoss(model=model)

    model.fit([(train_dataloader, train_loss)],
                epochs=3,
                output_path = f'models/knn/best_model',
                save_best_model = True,
                show_progress_bar = True,
                checkpoint_save_steps=12272,
                checkpoint_path = f'tmp/knn/'
                )
    
    model.save('models/knn/best_model/final/')

def inference():
    import math

    knn_model = SentenceTransformer('tmp/knn/24544')
    mnli_labels_emb = {}
    for mnli_label in mnli_labels:
        mnli_labels_emb[mnli_label] = torch.from_numpy(knn_model.encode(mnli_label))

    with open('data/test_text_nli.json') as input, \
        open(f'output/test_text_nli_probe_knn.json', 'w') as output:
        count = []
        for line in input.readlines()[:10]:
            line = json.loads(line)
            p = line['premise']
            h = line['hypothesis']
            label = line['label']
            p_emb = torch.from_numpy(knn_model.encode(p))
            h_emb = torch.from_numpy(knn_model.encode(h))
            min_dist = math.inf
            prediction = ''
            for mnli_label, mnli_label_emb in mnli_labels_emb.items():
                cur_dist = 1-F.cosine_similarity(torch.mul(h_emb, mnli_label_emb), p_emb, dim=0)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    prediction = mnli_label
            tmp = {'premise': p, 'hypothesis': h, 'prediction': prediction}
            output.write(json.dumps(tmp))
            output.write('\n')
            if prediction == 'entailment':
                prediction = '1'
            elif prediction == 'contradiction':
                prediction = '0'
            else:
                prediction = '-1'

            if prediction == label:
                count.append(1)
            else:
                count.append(0)
        
        print("total:", len(count))
        print("correct:", sum(count))
        print("incorrect:", len(count) - sum(count))
        print("accuracy:", round((sum(count) / len(count)),4))

    with open('data/test_delex_text_nli.json') as input, \
        open(f'output/test_delex_text_nli_probe_knn.json', 'w') as output:
        count = []
        for line in input.readlines()[:10]:
            line = json.loads(line)
            p = line['premise']
            h = line['hypothesis']
            label = line['label']
            p_emb = torch.from_numpy(knn_model.encode(p))
            h_emb = torch.from_numpy(knn_model.encode(h))
            min_dist = math.inf
            prediction = ''
            for mnli_label, mnli_label_emb in mnli_labels_emb.items():
                cur_dist = 1-F.cosine_similarity(torch.mul(h_emb, mnli_label_emb), p_emb, dim=0)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    prediction = mnli_label
            tmp = {'premise': p, 'hypothesis': h, 'prediction': prediction}
            output.write(json.dumps(tmp))
            output.write('\n')
            if prediction == 'entailment':
                prediction = '1'
            elif prediction == 'contradiction':
                prediction = '0'
            else:
                prediction = '-1'
                
            if prediction == label:
                count.append(1)
            else:
                count.append(0)
        
        print("total:", len(count))
        print("correct:", sum(count))
        print("incorrect:", len(count) - sum(count))
        print("accuracy:", round((sum(count) / len(count)),4))

inference()