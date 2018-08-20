#!/usr/bin/env python3
'''Load the Word2vec pretrained embedding.
'''

import logging
import os
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.keyedvectors import Vocab
from numpy import ndarray

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import TrainingData

LOGGER = logging.getLogger(__name__)


class EmbeddingDomain(Enum):
    '''Embedding domains.
    '''
    general = auto()
    real_estate = auto()


class Word2vecEmbeddingLoader(Component):
    '''Load the word2vec format embedding.
    '''
    name: str = "embedding_loader_word2vec"

    provides: List[str] = ['lookup_table']

    defaults: Dict[str, Any] = {'binary': 'false'}

    def __init__(self, component_config: Dict[str, Any],
                 lookup_table: KeyedVectors, domain: EmbeddingDomain) -> None:
        super(Word2vecEmbeddingLoader, self).__init__(component_config)
        self.lookup_table: KeyedVectors = lookup_table
        self.domain: EmbeddingDomain = domain

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['gensim']

    @classmethod
    def create(cls, cfg: RasaNLUModelConfig) -> 'Word2vecEmbeddingLoader':
        component_config: Dict[str, Any] = cfg.for_component(cls.name,
                                                             cls.defaults)
        file_path: str = component_config['file_path']
        is_binary: str = component_config.get('binary', 'false').lower()
        binary: bool = (is_binary == 'true')
        if __debug__:
            limit: Optional[int] = 5000
        else:
            limit = None
        lookup_table: KeyedVectors = KeyedVectors.load_word2vec_format(
            file_path, binary=binary, limit=limit)

        domain_str: str = component_config['domain']
        domain: EmbeddingDomain = EmbeddingDomain[domain_str]
        return cls(component_config, lookup_table, domain)

    @classmethod
    def cache_key(cls, model_metadata: Metadata) -> str:
        component_meta: Dict[str, Any] = model_metadata.for_component(cls.name)
        file_path: str = component_meta['file_path']
        return cls.name + file_path

    def provide_context(self) -> Dict[str, Any]:
        return {'lookup_table': self.generate_emb_matrix()}

    def generate_emb_matrix(self) -> ndarray:
        '''Generate a numpy matrix from the lookup table.
        '''
        vector_size: int = self.lookup_table.vector_size
        vocab_len: int = len(self.lookup_table.vocab)
        lookup_table_matrix = np.zeros((vocab_len, vector_size))
        for key, key_vocab in self.lookup_table.vocab.items():
            vec = self.lookup_table[key]
            lookup_table_matrix[key_vocab.index] = vec
        return lookup_table_matrix

    def persist(self, model_dir: str) -> Dict[str, Any]:
        lookup_table_path = os.path.join(model_dir, 'word2vec.bin')
        self.lookup_table.save_word2vec_format(lookup_table_path, binary=True)
        return {'lookup_table_path': lookup_table_path,
                'domain': self.domain.name}

    @classmethod
    def load(cls, model_dir: Optional[str]=None,
             model_metadata: Optional[Metadata]=None,
             cached_component: Optional['Word2vecEmbeddingLoader']=None,
             **kwargs: Any) -> 'Word2vecEmbeddingLoader':
        if cached_component:
            return cached_component
        if model_metadata is None:
            raise ValueError('No Metadata Loaded.')
        else:
            component_config = model_metadata.for_component(cls.name)
            lookup_table_path = component_config['lookup_table_path']
            lookup_table = KeyedVectors.load_word2vec_format(lookup_table_path,
                                                             binary=True)
            domain_str = component_config['domain']
            domain = EmbeddingDomain[domain_str]
        return cls(component_config, lookup_table, domain)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        for example in training_data.training_examples:
            tokens = example.get('tokens')
            example.set('token_ix_seq', self.sentence2ix_seq(tokens))

    def process(self, message: Message, **kwargs: Any) -> None:
        tokens = message.get('tokens')
        message.set('token_ix_seq', self.sentence2ix_seq(tokens))

    def sentence2ix_seq(self, tokens: List[Token]) -> List[int]:
        '''Convert sentence to a sequence of token indices.
        '''
        vocab: Dict[str, Vocab] = self.lookup_table.vocab
        ix_seq: List[int] = []
        # TODO: handle oov words
        for token in tokens:
            if token.text in vocab:
                ix_seq.append(vocab[token.text].index+1)
            else:
                ix_seq.append(0)
        return ix_seq


if __name__ == '__main__':
    pass
