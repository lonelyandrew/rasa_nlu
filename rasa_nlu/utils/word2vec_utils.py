#!/usr/bin/env python3
'''Load the Word2vec pretrained embedding.
'''

import logging
import os
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import json

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

    def __init__(self, *, component_config: Dict[str, Any],
                 domain: EmbeddingDomain=EmbeddingDomain.general,
                 lookup_table: Optional[KeyedVectors]=None,
                 vocab: Optional[Dict[str, Vocab]]=None) -> None:
        super(Word2vecEmbeddingLoader, self).__init__(component_config)
        self.lookup_table: KeyedVectors = lookup_table
        self.domain: EmbeddingDomain = domain
        self.oov_set: Set[str] = set()
        if lookup_table is not None:
            self.vocab: Dict[str, Vocab] = lookup_table.vocab
        else:
            if vocab is None:
                raise ValueError('Please offser at least a vocabulary or'
                                 'a lookup table.')
            else:
                self.vocab = vocab

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
            LOGGER.info(f'You are in DEBUG mode, load word embedding with limit count {limit}.')  # noqa
        else:
            limit = None
        lookup_table: KeyedVectors = KeyedVectors.load_word2vec_format(
            file_path, binary=binary, limit=limit)

        domain_str: str = component_config['domain']
        domain: EmbeddingDomain = EmbeddingDomain[domain_str]
        return cls(component_config=component_config, domain=domain,
                   lookup_table=lookup_table)

    @classmethod
    def cache_key(cls, model_metadata: Metadata) -> str:
        component_meta: Dict[str, Any] = model_metadata.for_component(cls.name)
        file_path: str = component_meta['file_path']
        return cls.name + file_path

    def provide_context(self) -> Dict[str, Any]:
        if self.lookup_table is not None:
            return {'lookup_table': self.generate_emb_matrix()}
        else:
            return {}

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
        vocab_path = os.path.join(model_dir, 'vocab.json')
        vocab_dict = self.vocab2dict(self.vocab)

        with open(vocab_path, 'w+') as f:
            json.dump(vocab_dict, f)
        return {'vocab_path': vocab_path,
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
            vocab_path = component_config['vocab_path']
            with open(vocab_path) as f:
                vocab_dict = json.load(f)
                vocab: Dict[str, Vocab] = cls.dict2vocab(vocab_dict)
            domain_str = component_config['domain']
            domain = EmbeddingDomain[domain_str]
        return cls(component_config=component_config, domain=domain,
                   lookup_table=None, vocab=vocab)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        for example in training_data.training_examples:
            tokens = example.get('tokens')
            example.set('token_ix_seq', self.sentence2ix_seq(tokens))
        LOGGER.info(self.oov_set)

    def process(self, message: Message, **kwargs: Any) -> None:
        tokens = message.get('tokens')
        message.set('token_ix_seq', self.sentence2ix_seq(tokens))

    @staticmethod
    def vocab2dict(vocab: Dict[str, Vocab]) -> Dict[str, int]:
        new_vocab = {}
        for k, v in vocab.items():
            new_vocab[k] = v.index
        return new_vocab

    @staticmethod
    def dict2vocab(vocab_dict: Dict[str, int]) -> Dict[str, Vocab]:
        vocab = {}
        for k, v in vocab_dict.items():
            vocab[k] = Vocab(index=v)
        return vocab

    def sentence2ix_seq(self, tokens: List[Token]) -> List[int]:
        '''Convert sentence to a sequence of token indices.
        '''
        ix_seq: List[int] = []
        token_seq: List[str] = []
        sentence = '-'.join([t.text for t in tokens])
        LOGGER.info('=' * 80)
        LOGGER.info(f'Processing: ' + sentence)
        for token in tokens:
            if not token.text:
                continue
            if token.text in self.oov_set:
                continue
            elif token.text in self.vocab:
                ix_seq.append(self.vocab[token.text].index+1)
                token_seq.append(token.text)
            else:
                LOGGER.info(f'NEW OOV TOKEN: "{token.text}"')
                for char in token.text:
                    if char in self.vocab:
                        LOGGER.info(f'ADD CHAR: "{char}"')
                        ix_seq.append(self.vocab[char].index+1)
                        token_seq.append(char)
                    else:
                        self.oov_set.add(char)
                        LOGGER.info(f'NEW OOV CHAR: "{char}"')
        LOGGER.info('ACTUAL TOKEN: ' + '-'.join(token_seq))
        if not ix_seq:
            ix_seq = [0]
        return ix_seq


if __name__ == '__main__':
    pass
