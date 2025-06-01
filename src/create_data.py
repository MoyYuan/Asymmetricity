from core.data import RelationDataProcessor, NLIDataProcessor

def main():
    data_dir = 'data'
    relation_processor = RelationDataProcessor(data_dir)
    nli_processor = NLIDataProcessor(data_dir)

    # Example usages:
    # Filter relations
    relation_processor.filter_relations('wikidata5m/wikidata5m_relation.txt', 'relation.txt')

    # Create train/test data
    relation_processor.create_data('wikidata5m/wikidata5m_transductive_all.txt', 'train.json', 'test.json')

    # Create id to title mapping
    relation_processor.create_id_to_title('wikidata5m/wikidata5m_entity.txt', 'id_to_title.pkl')

    # Prepare data for training
    relation_processor.prepare_data_for_training('prompts.txt', 'id_to_title.pkl')

    # NLI data creation/preparation
    nli_processor.create_nli_data('wikidata5m/wikidata5m_transductive_all.txt', 'train_nli.json', 'test_nli.json')
    nli_processor.prepare_nli_data_for_training('prompts.txt', 'id_to_title.pkl')

    # Count labels
    relation_processor.count_labels('train.json', 'test.json')

if __name__ == '__main__':

    for split in ['train', 'test']:
        with open(f'data/{split}_nli.json') as input,\
            open(f'data/{split}_text_nli.json', 'w') as output,\
            open(f'data/{split}_delex_text_nli.json', 'w') as delex_output:
                for line in input.readlines():
                    try:
                        line = json.loads(line)
                        rel = line['p']['rel']

                        p_subj = line['p']['subj']
                        p_obj = line['p']['obj']

                        h_subj = line['h']['subj']
                        h_obj = line['h']['obj']

                        label = line['label']

                        p_subj_text = id_to_text[p_subj]
                        p_obj_text = id_to_text[p_obj]

                        h_subj_text = id_to_text[h_subj]
                        h_obj_text = id_to_text[h_obj]

                        p_delex_text = prompts[rel].replace('[X]', p_subj).replace('[Y]', p_obj)
                        h_delex_text = prompts[rel].replace('[X]', h_subj).replace('[Y]', h_obj)

                        p_text = prompts[rel].replace('[X]', p_subj_text).replace('[Y]', p_obj_text)
                        h_text = prompts[rel].replace('[X]', h_subj_text).replace('[Y]', h_obj_text)

                        tmp = {'premise': f'{p_text}', 'hypothesis': f'{h_text}', 'label': f'{label}'}
                        output.write(json.dumps(tmp))
                        output.write('\n')

                        tmp = {'premise': f'{p_delex_text}', 'hypothesis': f'{h_delex_text}', 'label': f'{label}'}
                        delex_output.write(json.dumps(tmp))
                        delex_output.write('\n')
                    except:
                        continue

def count():
    count = []
    with open(f'data/train.txt') as f:
        for line in f.readlines():
            line = json.loads(line)
            count.append(int(line['label']))
    print('1', sum(count))
    print('0', len(count) - sum(count))

    count = []
    with open(f'data/test.txt') as f:
        for line in f.readlines():
            line = json.loads(line)
            count.append(int(line['label']))
    print('1', sum(count))
    print('0', len(count) - sum(count))

def main():
    create_nli_data()

if __name__ == '__main__':
    main()