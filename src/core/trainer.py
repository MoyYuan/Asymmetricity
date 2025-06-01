from abc import ABC, abstractmethod

class TrainerBase(ABC):
    """Abstract base class for training and experiment management."""
    def __init__(self, model, data_processor, config):
        self.model = model
        self.data_processor = data_processor
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class HuggingFaceTrainer(TrainerBase):
    def train(self):
        # Implement training loop using HuggingFace
        pass
    def evaluate(self):
        # Implement evaluation using HuggingFace
        pass

from torch.utils.data import DataLoader
import torch
import os

class CustomTrainer(TrainerBase):
    def train(self, train_examples=None, batch_size=32, epochs=1, save_path=None):
        self.model.build()
        if train_examples is None:
            raise ValueError('train_examples must be provided for custom training')
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.model.model.parameters(), lr=2e-5)
        self.model.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                # batch should be a dict or InputExample; adapt as needed
                sentence_features = batch['sentence_features']
                labels = batch['labels']
                optimizer.zero_grad()
                loss = self.model.forward(sentence_features, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}")
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.model.model.save(save_path)

    def evaluate(self, eval_examples=None, batch_size=32, metric_fn=None):
        self.model.model.eval()
        eval_dataloader = DataLoader(eval_examples, batch_size=batch_size)
        results = []
        with torch.no_grad():
            for batch in eval_dataloader:
                sentence_features = batch['sentence_features']
                labels = batch['labels']
                outputs = self.model.forward(sentence_features, labels)
                results.append(outputs.item())
        if metric_fn:
            return metric_fn(results)
        return results
