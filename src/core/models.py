from abc import ABC, abstractmethod

class BaseRelationModel(ABC):
    """Abstract base class for relation classification models."""
    def __init__(self, model_name_or_path, config=None):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.model = None
        self.tokenizer = None
        self.loss_fn = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

from sentence_transformers import SentenceTransformer
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Iterable, Dict

class ROTATELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 1):
        super(ROTATELoss, self).__init__()
        self.model = model
        self.margin = margin

    def get_config_dict(self):
        return {}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        h, p, r = reps
        distances = 1-F.cosine_similarity(torch.mul(h, r), p)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean()

class ROTATECosLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 1):
        super(ROTATECosLoss, self).__init__()
        self.model = model
        self.margin = margin

    def get_config_dict(self):
        return {}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        h, p, r = reps
        distances = torch.norm((torch.mul(h, r) - p), dim=1)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean()

class RotateRelationModel(BaseRelationModel):
    def __init__(self, model_name_or_path, config=None, margin=1):
        super().__init__(model_name_or_path, config)
        self.margin = margin
        self.model = None
        self.loss_fn = None

    def build(self):
        self.model = SentenceTransformer(self.model_name_or_path)
        self.loss_fn = ROTATELoss(self.model, self.margin)

    def forward(self, sentence_features, labels):
        return self.loss_fn(sentence_features, labels)

class KNNRelationModel(BaseRelationModel):
    def build(self):
        # Build KNN model
        pass
    def forward(self, *args, **kwargs):
        # Forward pass for KNN
        pass
