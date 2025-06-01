from transformers import pipeline
import json

model_name = "roberta-large-mnli"
model_name_short = model_name.split('/')[-1]
oracle = pipeline(model=model_name, device=0)

example = "albert moss (cricketer) is a sibling of Rudolf Mosse. </s></s> Rudolf Mosse is a sibling of albert moss (cricketer)."

with open('data/test_text_nli.json') as input, \
    open(f'output/test_text_nli_probe_{model_name_short}.json', 'w') as output:
    count = []
    for line in input.readlines():
        line = json.loads(line)
        p = line['premise']
        h = line['hypothesis']
        label = line['label']
        result = oracle(f"{p} </s></s> {h}", )       
        prediction = result[0]['label']
        tmp = {'premise': p, 'hypothesis': h, 'prediction': prediction}
        output.write(json.dumps(tmp))
        output.write('\n')
        if prediction == 'ENTAILMENT':
            prediction = '1'
        elif prediction == 'CONTRADICTION':
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
    open(f'output/test_delex_text_nli_probe_{model_name_short}.json', 'w') as output:
    count = []
    for line in input.readlines():
        line = json.loads(line)
        p = line['premise']
        h = line['hypothesis']
        label = line['label']
        result = oracle(f"{p} </s></s> {h}", )       
        prediction = result[0]['label']
        tmp = {'premise': p, 'hypothesis': h, 'prediction': prediction}
        output.write(json.dumps(tmp))
        output.write('\n')
        if prediction == 'ENTAILMENT':
            prediction = '1'
        elif prediction == 'CONTRADICTION':
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
