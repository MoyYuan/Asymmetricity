# Asymmetricity: OOP Modular Relation Classification Toolkit

## Overview

This project provides a modular, object-oriented framework for relation classification and probing tasks, with a focus on distinguishing symmetric and asymmetric relations using transformer models (e.g., BERT, RoBERTa, SentenceTransformers). The codebase is refactored for extensibility, maintainability, and research flexibility.

- **Core features:**
  - Modular OOP design for data processing, model definition, training, and evaluation
  - Support for custom loss functions (ROTATE, KNN, etc.)
  - Easy integration of new models, trainers, and evaluators
  - Clean separation of concerns for research and experimentation

---

## Project Structure

```
Asymmetricity/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Config class
│   │   ├── data.py           # DataProcessor classes
│   │   ├── evaluator.py      # Evaluator classes (RelationEvaluator, NLIEvaluator)
│   │   ├── models.py         # Model and loss classes (RotateRelationModel, KNNRelationModel, etc.)
│   │   └── trainer.py        # Trainer classes (CustomTrainer, HuggingFaceTrainer)
│   ├── asym_train.py         # OOP pipeline for relation classification
│   ├── create_data.py        # OOP data creation/processing entry point
│   ├── knn_llm.py            # OOP KNN training/evaluation entry point
│   ├── probe.py              # OOP probing/evaluation entry point
│   └── train.py              # OOP HuggingFace Trainer entry point
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Asymmetricity
   ```
2. **Install dependencies:**
   (Recommended: use a virtual environment)
   ```bash
   pip install -r requirements.txt
   ```
   - Main dependencies: `torch`, `transformers`, `sentence-transformers`, `datasets`, `numpy`, etc.

---

## Usage

### 1. Data Preparation
Run the OOP data processor to create and preprocess datasets:
```bash
python src/create_data.py
```

### 2. Training & Evaluation

- **Custom ROTATE model:**
  ```bash
  python src/asym_train.py
  ```
- **KNN model:**
  ```bash
  python src/knn_llm.py
  ```
- **Probing (NLI):**
  ```bash
  python src/probe.py
  ```
- **HuggingFace Trainer pipeline:**
  ```bash
  python src/train.py
  ```

### 3. Extending the Framework
- Add new data processors in `core/data.py`
- Define new models/losses in `core/models.py`
- Implement new trainers in `core/trainer.py`
- Add custom evaluators in `core/evaluator.py`

---

## Example: Custom Pipeline
Here is a typical pipeline using the OOP classes:

```python
from core.data import RelationDataProcessor
from core.models import RotateRelationModel
from core.trainer import CustomTrainer
from core.evaluator import RelationEvaluator

# Data
processor = RelationDataProcessor('data')
train_examples = processor.load_train_examples()

# Model
model = RotateRelationModel('roberta-large')

# Training
trainer = CustomTrainer(model, processor, config=None)
trainer.train(train_examples)

# Evaluation
relation_evaluator = RelationEvaluator(model, processor)
relation_evaluator.evaluate(model, train_examples)
```

---

## License
This project is licensed under the Apache 2.0 License.
