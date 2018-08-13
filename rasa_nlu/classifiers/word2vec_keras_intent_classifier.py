#!/usr/bin/env python3

'''Word2vec Keras Intent Classifier.
'''

from typing import List, Any, Dict
from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData
from rasa_nlu.config import RasaNLUModelConfig
from keras.models import import Model

class Word2vecKerasIntentClassifier(Component):
    '''Word2vec Keras Intent Classifier.
    '''
    name: str = 'intent_classifier_word2vec_keras'

    provides: List[str] = ['intent']

    requires: List[str] = ['token', 'lookup_table']

    def __init__(self, component_config: Dict[str, Any], clf: Model,
                 clf_config: Dict[str, Any]) -> None:
        super(Word2vecKerasIntentClassifier, self).__init__(component_config)
        self.clf: Model = clf
        self.clf_config = clf_config

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['keras']

    def train(self, training_data :TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        lookup_table = kwargs['lookup_table']


if __name__ == '__main__':
    pass
