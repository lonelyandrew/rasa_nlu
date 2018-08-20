#!/usr/bin/env python3

from typing import Any, Dict, Optional

from keras.layers import LSTM, Bidirectional, Dense, Embedding, Input, Masking
from keras.models import Model
from keras import optimizers
from tensorflow import Tensor


class KerasBaseModel:

    def __init__(self, clf_config: Dict[str, Any],
                 model: Optional[Model]) -> None:
        self.clf_config: Dict[str, Any] = clf_config
        self.model: Optional[model] = model

    def compile(self):
        raise NotImplementedError()


if __name__ == '__main__':
    pass
