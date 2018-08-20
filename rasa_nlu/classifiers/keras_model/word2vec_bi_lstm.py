#!/usr/bin/env python3

from typing import Dict, Any

from keras.layers import LSTM, Bidirectional, Dense, Embedding, Input, Masking
from keras.models import Model
from keras import optimizers
from tensorflow import Tensor
from rasa_nlu.classifiers.keras_model.keras_base_model import KerasBaseModel
from numpy import ndarray


class Word2vecIntentClassifier(KerasBaseModel):

    def __init__(self, clf_config: Dict[str, Any], lookup_table: ndarray,
                 nlabels: int) -> None:
        super(Word2vecIntentClassifier, self).__init__(clf_config, None)
        input_dim: int
        output_dim: int
        input_dim, output_dim = lookup_table.shape
        self.embedding_layer: Embedding = Embedding(input_dim, output_dim,
                                                    name='word_emb',
                                                    weights=[lookup_table],
                                                    mask_zero=True)
        lstm_layer: LSTM = LSTM(clf_config['hidden_size'], name='lstm')
        self.bilstm: Bidirectional = Bidirectional(lstm_layer,
                                                   merge_mode='concat',
                                                   name='bilstm')
        self.fc: Dense = Dense(nlabels, activation='softmax', name='fc')

    def compile(self):
        if self.model is None:
            token_input: Input = Input(shape=(None, ), dtype='int32')
            emb_out: Tensor = self.embedding_layer(token_input)
            mask_out: Tensor = Masking()(emb_out)
            bi_lstm_out: Tensor = self.bilstm(mask_out)
            fc_out: Tensor = self.fc(bi_lstm_out)
            model = Model(token_input, fc_out)
            optimizer_config = {'class_name': self.clf_config['optimizer'],
                                'config': {'lr': self.clf_config['lr']}}
            optimizer = optimizers.get(optimizer_config)
            model.compile(optimizer=optimizer, loss=self.clf_config['loss'])
            self.model = model


if __name__ == '__main__':
    pass
