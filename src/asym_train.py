from core.data import RelationDataProcessor
from core.models import RotateRelationModel
from core.trainer import CustomTrainer
from core.evaluator import RelationEvaluator
from core.config import Config
from sentence_transformers import InputExample
import torch
import random


def build_train_examples(data_processor, train_json_path):
    # Example: Load your train.json and convert to InputExample list
    import json
    train_examples = []
    with open(train_json_path) as f:
        for line in f:
            item = json.loads(line)
            # Adapt as needed for your data format
            text1 = item.get('subj', '')
            text2 = item.get('obj', '')
            label = item.get('label', 0)
            train_examples.append(InputExample(texts=[text1, text2], label=label))
    return train_examples


def main():
    # Config
    config = Config(model_name_or_path='bert-base-uncased', data_dir='data', margin=1)
    
    # Data
    data_processor = RelationDataProcessor(config.data_dir)
    train_examples = build_train_examples(data_processor, train_json_path=f"{config.data_dir}/train.json")
    # You may build eval_examples similarly
    eval_examples = build_train_examples(data_processor, train_json_path=f"{config.data_dir}/test.json")

    # Model
    model = RotateRelationModel(config.model_name_or_path, margin=config.margin)

    # Trainer
    trainer = CustomTrainer(model, data_processor, config)
    trainer.train(train_examples=train_examples, batch_size=32, epochs=3, save_path='models/rotate/best_model')

    # Evaluator (example, assumes model/model.encode is compatible)
    evaluator = RelationEvaluator(model, data_processor)
    # evaluator.evaluate(model.model, eval_examples, mode='ordered') # Uncomment/adapt as needed

if __name__ == '__main__':
    main()