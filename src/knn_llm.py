from core.models import KNNRelationModel
from core.trainer import CustomTrainer
from core.evaluator import NLIEvaluator
from core.data import NLIDataProcessor

def main():
    data_dir = 'data'
    processor = NLIDataProcessor(data_dir)
    model = KNNRelationModel('roberta-large')
    trainer = CustomTrainer(model, processor, config=None)
    evaluator = NLIEvaluator(model, processor)

    train_examples = processor.load_train_examples()
    trainer.train(train_examples)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
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