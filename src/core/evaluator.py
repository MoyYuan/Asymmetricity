from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Abstract base class for evaluation logic."""
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

    @abstractmethod
    def evaluate(self):
        pass

import numpy as np

class RelationEvaluator(Evaluator):
    def ordered_eval(self, model, dataset):
        Y_true = []
        Y_pred = []
        for i in dataset:
            sent1 = i['sentence1']
            sent2 = i['sentence2']
            rte_label = i['label']
            embeddings1 = model.encode(sent1, convert_to_tensor=True)
            embeddings2 = model.encode(sent2, convert_to_tensor=True)
            distance = model.ORDERED_COSINE_DISTANCE(embeddings1.reshape(1,-1), embeddings2.reshape(1,-1)).item()
            Y_true.append(rte_label)
            Y_pred.append(distance)
        mse = np.square(np.subtract(Y_true, Y_pred)).mean()
        print(f"Ordered MSE: {mse}")
        return mse

    def normal_eval(self, model, dataset):
        Y_true = []
        Y_pred = []
        for i in dataset:
            sent1 = i['sentence1']
            sent2 = i['sentence2']
            rte_label = i['label']
            embeddings1 = model.encode(sent1, convert_to_tensor=True)
            embeddings2 = model.encode(sent2, convert_to_tensor=True)
            distance = model.COSINE_DISTANCE(embeddings1.reshape(1,-1), embeddings2.reshape(1,-1)).item()
            Y_true.append(rte_label)
            Y_pred.append(distance)
        mse = np.square(np.subtract(Y_true, Y_pred)).mean()
        print(f"Normal MSE: {mse}")
        return mse

    def evaluate(self, model, dataset, mode='ordered'):
        if mode == 'ordered':
            return self.ordered_eval(model, dataset)
        else:
            return self.normal_eval(model, dataset)

class NLIEvaluator(Evaluator):
    def evaluate(self):
        # Implement evaluation for NLI tasks
        pass
