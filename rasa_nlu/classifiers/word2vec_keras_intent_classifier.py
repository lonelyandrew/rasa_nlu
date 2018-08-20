#!/usr/bin/env python3

'''Word2vec Keras Intent Classifier.
'''

import json
import os
import logging
import random
from typing import Any, Dict, Generator, List, Optional, Tuple
from types import ModuleType
from math import ceil
from importlib import import_module

import numpy as np
from numpy import ndarray
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
import progressbar
from progressbar import widgets

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Message, Metadata
from rasa_nlu.training_data import TrainingData
from rasa_nlu.classifiers.keras_model.keras_base_model import KerasBaseModel

LOGGER = logging.getLogger(__name__)

TrainingBatch = Generator[Tuple[ndarray, ndarray], None, None]
BarWidgetsReturn = Tuple[List[widgets.WidgetBase],
                         widgets.WidgetBase,
                         widgets.WidgetBase]


class Word2vecKerasIntentClassifier(Component):
    '''Word2vec Keras Intent Classifier.
    '''
    name: str = 'intent_classifier_word2vec_keras'

    provides: List[str] = ['intent']

    requires: List[str] = ['tokens', 'lookup_table']

    def __init__(self, component_config: Dict[str, Any],
                 clf_config: Dict[str, Any],
                 clf: Optional[KerasBaseModel]=None,
                 labels: Optional[List[str]]=None) -> None:
        super(Word2vecKerasIntentClassifier, self).__init__(component_config)
        self.clf_config: Dict[str, Any] = clf_config
        self.clf = clf
        self.labels = labels

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['keras']

    @staticmethod
    def load_clf_class_by_path(clf_class_path: str):
        clf_module_name: str = '.'.join(clf_class_path.split('.')[:-1])
        clf_class_name: str = clf_class_path.split('.')[-1]
        clf_module: ModuleType = import_module(clf_module_name)
        clf_class: type = getattr(clf_module, clf_class_name)
        return clf_class

    @staticmethod
    def get_progressbar_widget_list(nepoch: int,
                                    nbatches: int) -> BarWidgetsReturn:
        epoch_status_fmt_str: str = 'EPOCH: %(epoch_ix)d/%(nepoch)d'
        epoch_status = progressbar.FormatCustomText(epoch_status_fmt_str,
                                                    dict(epoch_ix=0,
                                                         nepoch=nepoch))
        batch_status_fmt_str: str = 'BATCH: %(batch_ix)d/%(nbatches)d'
        batch_status = progressbar.FormatCustomText(batch_status_fmt_str,
                                                    dict(batch_ix=0,
                                                         nbatches=nbatches))
        widgets_list: List[widgets.WidgetBase] = [
            widgets.Percentage(),
            ' ', epoch_status,
            ' ', batch_status,
            ' ', widgets.Bar(),
            ' ', widgets.Timer(),
        ]
        return widgets_list, epoch_status, batch_status

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        if self.clf is None:
            lookup_table: ndarray = kwargs['lookup_table']
            if self.labels is None:
                labels: List[str] = [e.get("intent")
                                     for e in training_data.intent_examples]
                self.labels: List[str] = list(set(labels))
            nlabels = len(set(self.labels))
            if nlabels < 2:
                raise ValueError('At lease two kinds of labels.')
            else:
                self.clf_config.update({'nlabels': nlabels})
            clf_class_path: str = self.clf_config['clf_class']
            clf_class = self.load_clf_class_by_path(clf_class_path)
            clf: KerasBaseModel = clf_class(self.clf_config, lookup_table,
                                            nlabels)
            clf.compile()
        batch_size = self.clf_config['batch_size']
        nepoch = self.clf_config['nepoch']
        nbatches = ceil(len(training_data.intent_examples) / batch_size)
        widgets_list, epoch_status, batch_status = self.get_progressbar_widget_list(nepoch, nbatches)  # noqa
        with progressbar.ProgressBar(max_value=nbatches,
                                     redirect_stdout=True,
                                     widgets=widgets_list) as bar:
            prev_epoch_loss = 0.0
            for epoch_ix in range(nepoch):
                epoch_loss = 0.0
                epoch_status.update_mapping(epoch_ix=epoch_ix+1)
                batches = self.generate_batch(training_data.intent_examples)
                for batch_ix, x_y in enumerate(batches):
                    batch_status.update_mapping(batch_ix=batch_ix+1)
                    bar.update(batch_ix+1)
                    batch_x, batch_y = x_y
                    epoch_loss += clf.model.train_on_batch(batch_x, batch_y)
                print(f'LOSS {epoch_ix}/{nepoch}: {epoch_loss:.4} ({epoch_loss-prev_epoch_loss:+.4})')  # noqa
                prev_epoch_loss = epoch_loss
                bar.update(0)
        self.clf = clf

    def process(self, message: Message, **kwargs: Any):
        intent: Optional[str] = None
        if self.clf is not None and self.labels is not None:
            input_x: ndarray = np.array([message.get('token_ix_seq')])
            pred = self.clf.model.predict(input_x)[0]
            intent_idx = pred.argmax()
            intent = self.labels[intent_idx]
        message.set('intent', intent, add_to_output=True)

    def generate_batch(self, examples: List[Message],
                       shuffle: bool=True) -> TrainingBatch:
        '''Group training examples into batches.
        '''
        if self.labels is None:
            raise ValueError('No Labels.')
        batch_size: int = self.clf_config['batch_size']
        unbatch_list: List[List[int]] = []
        y_list: List[int] = []
        if shuffle:
            random.shuffle(examples)
        nlabels: int = self.clf_config['nlabels']
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

    @classmethod
    def load(cls, model_dir: str, model_metadata: Metadata,
             cached_component: Component,
             **kwargs: Any) -> 'Word2vecKerasIntentClassifier':
        meta: Dict[str, Any] = model_metadata.for_component(cls.name)
        clf_config_file_path: str = meta['clf_config_file_path']
        with open(clf_config_file_path) as f:
            clf_config: Dict[str, Any] = json.load(f)
        clf_file_path: str = meta['clf_file_path']
        clf_model: Model = load_model(clf_file_path)
        clf: KerasBaseModel = KerasBaseModel(clf_config, clf_model)
        labels_file_path: str = meta['labels_file_path']
        with open(labels_file_path) as g:
            labels: List[str] = json.load(g)
        return cls(meta, clf_config, clf, labels)

    def persist(self, model_dir: str) -> Dict[str, Any]:
        clf_config_file_path: str = os.path.join(model_dir, 'clf_config.json')
        with open(clf_config_file_path, 'w+') as f:
            json.dump(self.clf_config, f)
        clf_file_path: str = ''
        if self.clf is not None:
            clf_file_path = os.path.join(model_dir, 'clf.h5')
            self.clf.model.save(clf_file_path)
        else:
            raise ValueError('Please Build the Classifier First.')
        labels_file_path: str = os.path.join(model_dir, 'labels.json')
        if self.labels is not None:
            with open(labels_file_path, 'w+') as g:
                json.dump(self.labels, g)
        return {'clf_config_file_path': clf_config_file_path,
                'clf_file_path': clf_file_path,
                'labels_file_path': labels_file_path}

    @classmethod
    def create(cls,
               cfg: RasaNLUModelConfig) -> 'Word2vecKerasIntentClassifier':
        component_config: Dict[str, Any] = cfg.for_component(cls.name)
        clf_config_file_path: str = component_config['clf_config_file_path']
        clf_file_path: Optional[str] = component_config.get('clf_file_path',
                                                            None)
        labels: Optional[List[str]] = component_config.get('labels', None)
        with open(clf_config_file_path) as f:
            clf_config: Dict[str, Any] = json.load(f)
        if clf_file_path is not None:
            clf_model: Model = load_model(clf_file_path)
            clf = KerasBaseModel(clf_config_file_path, clf_model)
        else:
            clf = None
        return cls(component_config, clf_config, clf, labels)


if __name__ == '__main__':
    pass
