#!/usr/bin/env python3

'''Word2vec Keras Intent Classifier.
'''

import json
import os
import logging
import random
from typing import Any, Dict, Generator, List, Optional, Tuple
from math import ceil

import numpy as np
from numpy import ndarray
from keras import optimizers
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Input, Masking
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from tensorflow import Tensor
import progressbar

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Message, Metadata
from rasa_nlu.training_data import TrainingData

LOGGER = logging.getLogger(__name__)

TrainingBatch = Generator[Tuple[ndarray, ndarray], None, None]


class Word2vecKerasIntentClassifier(Component):
    '''Word2vec Keras Intent Classifier.
    '''
    name: str = 'intent_classifier_word2vec_keras'

    provides: List[str] = ['intent']

    requires: List[str] = ['tokens', 'lookup_table']

    def __init__(self, component_config: Dict[str, Any],
                 clf_config: Dict[str, Any],
                 clf: Optional[Model]) -> None:
        super(Word2vecKerasIntentClassifier, self).__init__(component_config)
        self.clf_config: Dict[str, Any] = clf_config
        self.clf = clf

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['keras']

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        if self.clf is None:
            lookup_table: ndarray = kwargs['lookup_table']
            labels: List[str] = [e.get("intent")
                                 for e in training_data.intent_examples]
            self.labels: List[str] = list(set(labels))
            nlabels = len(set(self.labels))
            if nlabels < 2:
                raise ValueError('At lease two kinds of labels.')
            else:
                self.clf_config.update({'nlabels': nlabels})
            clf: Model = self.build_clf(lookup_table)
        # TODO: add epocs
        batch_size = self.clf_config['batch_size']
        nbatches = ceil(len(training_data.intent_examples) / batch_size)
        with progressbar.ProgressBar(max_value=nbatches,
                                     redirect_stdout=True) as bar:
            batch_ix = 1
            for batch_x, batch_y in self.generate_batch(
                    training_data.intent_examples):
                loss = clf.train_on_batch(batch_x, batch_y)
                print(loss)
                bar.update(batch_ix)
                batch_ix += 1

    def process(self, message: Message, **kwargs: Any):

        if not self.clf:
            intent = None
        else:
            input_x = np.array([message.get('token_ix_seq')])
            pred = self.clf.predict(input_x)[0]
            intent_idx = pred.argmax()
            intent = self.labels[intent_idx]
        message.set('intent', intent, add_to_output=True)

    def generate_batch(self, examples: List[Message],
                       shuffle: bool=True) -> TrainingBatch:
        '''Group training examples into batches.
        '''
        batch_size: int = self.clf_config['batch_size']
        unbatch_list: List[List[int]] = []
        y_list: List[int] = []
        if shuffle:
            random.shuffle(examples)
        nlabels = self.clf_config['nlabels']
        for example in examples:
            token_ix_seq: List[int] = example.get('token_ix_seq')
            unbatch_list.append(token_ix_seq)
            y_list.append(self.labels.index(example.get('intent')))
            if len(unbatch_list) == batch_size:
                batch_list: ndarray = self.pad_batch_list(unbatch_list)
                y_array: ndarray = to_categorical(y_list, num_classes=nlabels)
                unbatch_list = []
                y_list = []
                yield batch_list, y_array

    @staticmethod
    def pad_batch_list(unbatch_list: List[List[int]]) -> ndarray:
        '''Pad sequences to a fixed length.
        '''
        batch_list: List[List[int]] = []
        max_len: int = len(max(unbatch_list, key=len))
        for seq in unbatch_list:
            pad_len: int = max_len - len(seq)
            batch_list.append(seq + [0] * pad_len)
        return np.array(batch_list)

    def build_clf(self, lookup_table: ndarray) -> Model:
        '''Build the classifier.
        '''
        # TODO: abstract this part
        token_input: Input = Input(shape=(None, ), dtype='int32')
        input_dim, output_dim = lookup_table.shape
        embedding_layer: Embedding = Embedding(input_dim, output_dim,
                                               name='word_emb',
                                               weights=[lookup_table],
                                               mask_zero=True)
        emb_out: Tensor = embedding_layer(token_input)
        mask_out: Tensor = Masking()(emb_out)
        lstm_layer: LSTM = LSTM(self.clf_config['hidden_size'], name='lstm')
        bi_lstm_out: Tensor = Bidirectional(lstm_layer,
                                            merge_mode='concat',
                                            name='blstm')(mask_out)
        dense_out: Tensor = Dense(self.clf_config['nlabels'],
                                  activation='softmax',
                                  name='dense')(bi_lstm_out)
        clf = Model(token_input, dense_out)

        optimizer_config = {'class_name': self.clf_config['optimizer'],
                            'config': {'lr': self.clf_config['lr']}}
        optimizer = optimizers.get(optimizer_config)
        clf.compile(optimizer=optimizer, loss=self.clf_config['loss'])
        return clf

    @classmethod
    def load(cls, model_dir: str, model_metadata: Metadata,
             cached_component: Component,
             **kwargs: Any) -> 'Word2vecKerasIntentClassifier':
        meta: Dict[str, Any] = model_metadata.for_component(cls.name)
        clf_config_file_path: str = meta['clf_config_file_path']
        with open(clf_config_file_path) as f:
            clf_config: Dict[str, Any] = json.load(f)
        clf_file_path: str = meta['clf_file_path']
        clf: Model = load_model(clf_file_path)
        return cls(meta, clf_config, clf)

    def persist(self, model_dir: str) -> Dict[str, Any]:
        clf_config_file_path: str = os.path.join(model_dir, 'clf_config.json')
        with open(clf_config_file_path, 'w+') as f:
            json.dump(self.clf_config, f)
        clf_file_path: str = ''
        if self.clf is not None:
            clf_file_path = os.path.join(model_dir, 'clf.h5')
            self.clf.save(clf_file_path)
        return {'clf_config_file_path': clf_config_file_path,
                'clf_file_path': clf_file_path}

    @classmethod
    def create(cls,
               cfg: RasaNLUModelConfig) -> 'Word2vecKerasIntentClassifier':
        component_config: Dict[str, Any] = cfg.for_component(cls.name)
        clf_config_file_path: str = component_config['clf_config_file_path']
        clf_file_path: Optional[str] = component_config.get('clf_file_path',
                                                            None)
        with open(clf_config_file_path) as f:
            clf_config: Dict[str, Any] = json.load(f)
        if clf_file_path is not None:
            clf: Model = load_model(clf_file_path)
        else:
            clf = None
        return cls(component_config, clf_config, clf)


if __name__ == '__main__':
    pass
