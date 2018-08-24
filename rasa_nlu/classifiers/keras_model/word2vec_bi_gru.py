#!/usr/bin/env python3

from typing import Dict, Any
import logging

from keras.layers import GRU, Bidirectional, Dense, Embedding, Input, Masking
from keras.layers import Dropout
from keras.models import Model
from keras import optimizers
from tensorflow import Tensor
from rasa_nlu.classifiers.keras_model.keras_base_model import KerasBaseModel
from numpy import ndarray


LOGGER = logging.getLogger(__name__)


class Word2vecIntentClassifier(KerasBaseModel):

    def __init__(self, clf_config: Dict[str, Any], lookup_table: ndarray,
                 nlabels: int) -> None:
        super(Word2vecIntentClassifier, self).__init__(clf_config, None)
        LOGGER.info(f'KERAS CONFIG: {clf_config}')
        input_dim: int
        output_dim: int
        input_dim, output_dim = lookup_table.shape
        self.embedding_layer: Embedding = Embedding(input_dim, output_dim,
                                                    name='word_emb',
                                                    weights=[lookup_table],
                                                    mask_zero=True)
        lstm_layer = GRU(clf_config['hidden_size'], name='lstm')
        self.bilstm: Bidirectional = Bidirectional(lstm_layer,
                                                   merge_mode='concat',
                                                   name='bilstm')
        self.lstm_dropout: Dropout = Dropout(0.5)
        self.fc: Dense = Dense(nlabels, activation='softmax', name='fc')

    def compile(self):
        if self.model is None:
            token_input: Input = Input(shape=(None, ), dtype='int32')
            emb_out: Tensor = self.embedding_layer(token_input)
            mask_out: Tensor = Masking()(emb_out)
            bi_lstm_out: Tensor = self.bilstm(mask_out)
            dropout_out: Tensor = self.lstm_dropout(bi_lstm_out)
            fc_out: Tensor = self.fc(dropout_out)
            model = Model(token_input, fc_out)
            metrics = ['sparse_categorical_accuracy']
            optimizer_config = {'class_name': self.clf_config['optimizer'],
                                'config': {'lr': self.clf_config['lr']}}
            optimizer = optimizers.get(optimizer_config)
            model.compile(optimizer=optimizer, loss=self.clf_config['loss'],
                          metrics=metrics)
            self.model = model


if __name__ == '__main__':
    pass
