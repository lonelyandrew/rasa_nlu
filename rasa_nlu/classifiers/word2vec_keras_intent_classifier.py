#!/usr/bin/env python3

'''Word2vec Keras Intent Classifier.
'''

import json
import os
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple
from types import ModuleType
from importlib import import_module

import numpy as np
from numpy import ndarray
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing.sequence import pad_sequences
import progressbar
from progressbar import widgets

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Message, Metadata
from rasa_nlu.training_data import TrainingData
from rasa_nlu.classifiers.keras_model.keras_base_model import KerasBaseModel


LOGGER = logging.getLogger(__name__)

TrainingBatch = Generator[Tuple[ndarray, ndarray], None, None]
BarWidgetsReturn = Tuple[List[widgets.WidgetBase], widgets.WidgetBase,
                         widgets.WidgetBase]


class TrainingMonitor(Callback):

    def __init__(self, **kwargs):
        super(TrainingMonitor, self).__init__()

    @staticmethod
    def get_progressbar_widget_list(nepoch: int) -> BarWidgetsReturn:
        epoch_status_fmt_str: str = 'EPOCH: %(epoch_ix)d/%(nepoch)d'
        epoch_status = progressbar.FormatCustomText(epoch_status_fmt_str,
                                                    dict(epoch_ix=0,
                                                         nepoch=nepoch))
        widgets_list: List[widgets.WidgetBase] = [
            widgets.Percentage(),
            ' ', widgets.SimpleProgress(
                format='(%s)' % widgets.SimpleProgress.DEFAULT_FORMAT),
            ' ', epoch_status,
            ' ', widgets.Bar(),
            ' ', widgets.Timer(),
        ]
        return widgets_list, epoch_status

    def on_train_begin(self, logs):
        nepoch: int = self.params['epochs']
        widgets_list, epoch_status = self.get_progressbar_widget_list(nepoch)  # noqa
        nbatches = self.params['samples'] // self.params['batch_size']
        self.epoch_status = epoch_status
        self.bar = progressbar.ProgressBar(max_value=nbatches,
                                           redirect_stdout=True,
                                           widgets=widgets_list)
        self.bar.start()
        self.history = {'loss': [0], 'acc': [0]}
        self.min_loss = float('inf')
        self.min_loss_epoch = float('nan')
        self.max_acc = -float('inf')
        self.max_acc_epoch = float('nan')

    def on_epoch_begin(self, epoch, logs):
        epoch += 1
        self.epoch_status.update_mapping(epoch_ix=epoch)

    def on_epoch_end(self, epoch, logs):
        self.bar.update(self.bar.max_value, force=True)
        epoch += 1
        epoch_loss = logs['loss']
        epoch_acc = logs['sparse_categorical_accuracy']
        if epoch_acc > self.max_acc:
            self.max_acc = epoch_acc
            self.max_acc_epoch = epoch
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            self.min_loss_epoch = epoch
        epoch_header = f'EPOCH ({epoch}/{self.params["epochs"]})-'
        epoch_loss_diff = epoch_loss - self.history['loss'][-1]
        epoch_acc_diff = epoch_acc - self.history['acc'][-1]

        if epoch == 1:
            epoch_loss_diff = float('nan')

        epoch_loss_str = f'LOSS:{epoch_loss:.4}' \
            f' (MIN LOSS {self.min_loss:.4} @ EPOCH {self.min_loss_epoch})' \
            f' ({epoch_loss_diff:+.4})\t'
        epoch_acc_str = f'ACC:{epoch_acc:.4}' \
            f' (MAX ACC {self.max_acc:.4} @ EPOCH {self.max_acc_epoch})' \
            f' ({epoch_acc_diff:+.4})'
        LOGGER.info(f'\n{epoch_header}{epoch_loss_str} {epoch_acc_str}')
        self.history['loss'].append(epoch_loss)
        self.history['acc'].append(epoch_acc)

    def on_batch_begin(self, batch, logs):
        self.bar.update(batch)


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
            self.clf = clf
        batch_size = self.clf_config['batch_size']
        nepoch = self.clf_config['nepoch']
        sequence = [t.get('token_ix_seq') for t
                    in training_data.intent_examples]
        max_len = len(max(sequence, key=len))
        X = pad_sequences(sequence, maxlen=max_len, padding='post')
        y = [self.labels.index(t.get('intent')) for t
             in training_data.intent_examples]
        min_delta = self.clf_config['min_delta']
        patience = self.clf_config['patience']
        early_stopping_callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', min_delta=min_delta,
                patience=patience)
        monitor = TrainingMonitor()
        callbacks = [early_stopping_callback, monitor]
        self.clf.model.fit(X, y, batch_size=batch_size, verbose=0,
                           epochs=nepoch, callbacks=callbacks)

    def process(self, message: Message, **kwargs: Any):
        intent_name: Optional[str] = None
        intent_confidence: float = 0.0
        if self.clf is not None and self.labels is not None:
            input_x: ndarray = np.array([message.get('token_ix_seq')])

            # LOGGER.debug(message.text)
            LOGGER.debug(input_x.shape)
            LOGGER.debug('-'.join([t.text for t in message.get('tokens')]))
            pred = self.clf.model.predict(input_x)[0]
            intent_idx = pred.argmax()
            intent_name = self.labels[intent_idx]
            intent_confidence = pred[intent_idx]
        intent = {'name': intent_name, 'confidence': intent_confidence}
        message.set('intent', intent, add_to_output=True)

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
        LOGGER.info(f'CLASSIFIER CONFIG: {component_config}')
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
