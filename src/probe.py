from core.evaluator import NLIEvaluator
from core.models import KNNRelationModel
from core.data import NLIDataProcessor

def main():
    data_dir = 'data'
    processor = NLIDataProcessor(data_dir)
    model = KNNRelationModel('roberta-large-mnli')
    evaluator = NLIEvaluator(model, processor)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
