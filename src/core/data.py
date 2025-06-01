from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Abstract base class for data processing."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def prepare(self):
        pass

import json
import pickle
import random
import os

class RelationDataProcessor(DataProcessor):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.symmetric_relations = []
        self.asymmetric_relations = []
        self._load_relation_types()

    def _load_relation_types(self):
        with open(os.path.join(self.data_dir, 'symmetric_relations.txt')) as f:
            self.symmetric_relations = [line.strip() for line in f]
        with open(os.path.join(self.data_dir, 'asymmetric_relations.txt')) as f:
            self.asymmetric_relations = [line.strip() for line in f]

    def filter_relations(self, input_file, output_file):
        with open(os.path.join(self.data_dir, input_file)) as inp, \
             open(os.path.join(self.data_dir, output_file), 'w') as out:
            for line in inp:
                id = line.strip().split('\t')[0]
                if id in self.symmetric_relations or id in self.asymmetric_relations:
                    out.write(line)

    def create_data(self, input_file, train_file, test_file):
        random.seed(0)
        with open(os.path.join(self.data_dir, input_file)) as inp, \
             open(os.path.join(self.data_dir, train_file), 'w') as out_train, \
             open(os.path.join(self.data_dir, test_file), 'w') as out_test:
            for line in inp:
                subj, rel, obj = line.strip().split('\t')
                if rel in self.symmetric_relations:
                    tmp = {'subj': subj, 'rel': rel, 'obj': obj, 'label': 1}
                    out_train.write(json.dumps(tmp) + '\n')
                    tmp = {'subj': obj, 'rel': rel, 'obj': subj, 'label': 1}
                    if random.random() < 0.5:
                        out_test.write(json.dumps(tmp) + '\n')
                    else:
                        out_train.write(json.dumps(tmp) + '\n')
                elif rel in self.asymmetric_relations:
                    tmp = {'subj': subj, 'rel': rel, 'obj': obj, 'label': 1}
                    out_train.write(json.dumps(tmp) + '\n')
                    tmp = {'subj': obj, 'rel': rel, 'obj': subj, 'label': 0}
                    if random.random() < 0.04:
                        out_test.write(json.dumps(tmp) + '\n')
                    else:
                        out_train.write(json.dumps(tmp) + '\n')

    def create_id_to_title(self, entity_file, output_pkl):
        id_to_text = {}
        with open(os.path.join(self.data_dir, entity_file)) as f:
            for line in f:
                id, title = line.strip().split('\t')[:2]
                id_to_text[id] = title
        with open(os.path.join(self.data_dir, output_pkl), 'wb') as f:
            pickle.dump(id_to_text, f)

    def prepare_data_for_training(self, prompts_file, id_to_title_pkl, splits=['train', 'test']):
        with open(os.path.join(self.data_dir, prompts_file)) as f:
            prompts = {line.split('\t')[0]: line.split('\t')[1].strip() for line in f}
        with open(os.path.join(self.data_dir, id_to_title_pkl), 'rb') as f:
            id_to_text = pickle.load(f)
        for split in splits:
            with open(os.path.join(self.data_dir, f'{split}.json')) as inp, \
                 open(os.path.join(self.data_dir, f'{split}_text.json'), 'w') as out, \
                 open(os.path.join(self.data_dir, f'{split}_delex_text.json'), 'w') as delex_out:
                for line in inp:
                    try:
                        line = json.loads(line)
                        subj, rel, obj, label = line['subj'], line['rel'], line['obj'], line['label']
                        subj_text = id_to_text[subj]
                        obj_text = id_to_text[obj]
                        delex_text = prompts[rel].replace('[X]', subj).replace('[Y]', obj)
                        text = prompts[rel].replace('[X]', subj_text).replace('[Y]', obj_text)
                        out.write(json.dumps({'text': text, 'label': label}) + '\n')
                        delex_out.write(json.dumps({'text': delex_text, 'label': label}) + '\n')
                    except Exception:
                        continue

    def count_labels(self, train_file, test_file):
        for split_file in [train_file, test_file]:
            count = []
            with open(os.path.join(self.data_dir, split_file)) as f:
                for line in f:
                    line = json.loads(line)
                    count.append(int(line['label']))
            print(f'{split_file} - 1s: {sum(count)}, 0s: {len(count) - sum(count)}')

class NLIDataProcessor(DataProcessor):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.symmetric_relations = []
        self.asymmetric_relations = []
        self._load_relation_types()

    def _load_relation_types(self):
        with open(os.path.join(self.data_dir, 'symmetric_relations.txt')) as f:
            self.symmetric_relations = [line.strip() for line in f]
        with open(os.path.join(self.data_dir, 'asymmetric_relations.txt')) as f:
            self.asymmetric_relations = [line.strip() for line in f]

    def create_nli_data(self, input_file, train_file, test_file):
        random.seed(0)
        with open(os.path.join(self.data_dir, input_file)) as inp, \
             open(os.path.join(self.data_dir, train_file), 'w') as out_train, \
             open(os.path.join(self.data_dir, test_file), 'w') as out_test:
            for line in inp:
                subj, rel, obj = line.strip().split('\t')
                if rel in self.symmetric_relations:
                    tmp = {'p': {'subj': subj, 'rel': rel, 'obj': obj}, 'h': {'subj': obj, 'rel': rel, 'obj': subj}, 'label': 1}
                    out_train.write(json.dumps(tmp) + '\n')
                    tmp = {'p': {'subj': obj, 'rel': rel, 'obj': subj}, 'h': {'subj': subj, 'rel': rel, 'obj': obj}, 'label': 1}
                    if random.random() < 0.5:
                        out_test.write(json.dumps(tmp) + '\n')
                    else:
                        out_train.write(json.dumps(tmp) + '\n')
                elif rel in self.asymmetric_relations:
                    tmp = {'p': {'subj': subj, 'rel': rel, 'obj': obj}, 'h': {'subj': obj, 'rel': rel, 'obj': subj}, 'label': 0}
                    out_train.write(json.dumps(tmp) + '\n')
                    tmp = {'p': {'subj': obj, 'rel': rel, 'obj': subj}, 'h': {'subj': subj, 'rel': rel, 'obj': obj}, 'label': 0}
                    if random.random() < 0.04:
                        out_test.write(json.dumps(tmp) + '\n')
                    else:
                        out_train.write(json.dumps(tmp) + '\n')

    def prepare_nli_data_for_training(self, prompts_file, id_to_title_pkl, splits=['train', 'test']):
        with open(os.path.join(self.data_dir, prompts_file)) as f:
            prompts = {line.split('\t')[0]: line.split('\t')[1].strip() for line in f}
        with open(os.path.join(self.data_dir, id_to_title_pkl), 'rb') as f:
            id_to_text = pickle.load(f)
        for split in splits:
            with open(os.path.join(self.data_dir, f'{split}_nli.json')) as inp, \
                 open(os.path.join(self.data_dir, f'{split}_text_nli.json'), 'w') as out, \
                 open(os.path.join(self.data_dir, f'{split}_delex_text_nli.json'), 'w') as delex_out:
                for line in inp:
                    try:
                        line = json.loads(line)
                        rel = line['p']['rel']
                        p_subj, p_obj = line['p']['subj'], line['p']['obj']
                        h_subj, h_obj = line['h']['subj'], line['h']['obj']
                        label = line['label']
                        p_subj_text = id_to_text[p_subj]
                        p_obj_text = id_to_text[p_obj]
                        h_subj_text = id_to_text[h_subj]
                        h_obj_text = id_to_text[h_obj]
                        p_delex_text = prompts[rel].replace('[X]', p_subj).replace('[Y]', p_obj)
                        h_delex_text = prompts[rel].replace('[X]', h_subj).replace('[Y]', h_obj)
                        p_text = prompts[rel].replace('[X]', p_subj_text).replace('[Y]', p_obj_text)
                        h_text = prompts[rel].replace('[X]', h_subj_text).replace('[Y]', h_obj_text)
                        out.write(json.dumps({'premise': p_text, 'hypothesis': h_text, 'label': label}) + '\n')
                        delex_out.write(json.dumps({'premise': p_delex_text, 'hypothesis': h_delex_text, 'label': label}) + '\n')
                    except Exception:
                        continue
