#!/usr/bin/env python3
'''Load the Word2vec pretrained embedding.
'''

import os
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from gensim.models.keyedvectors import KeyedVectors

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata

from rasa_nlu.tokenizers import Token

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
        self.lookup_table = lookup_table
        self.domain: EmbeddingDomain = domain

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['gensim']

    @classmethod
    def create(cls, cfg:RasaNLUModelConfig) -> 'Word2vecEmbeddingLoader':
        component_config: Dict[str, Any] = cfg.for_component(cls.name,
                                                             cls.defaults)
        file_path: str = component_config['file_path']
        is_binary: str = component_config.get('binary', 'false').lower()
        binary: bool = (is_binary == 'true')
        lookup_table: KeyedVectors = KeyedVectors.load_word2vec_format(
            file_path, binary=binary)

        domain_str: str = component_config['domain']
        domain: EmbeddingDomain = EmbeddingDomain[domain_str]
        return cls(component_config, lookup_table, domain)

    @classmethod
    def cache_key(cls, model_metadata: Metadata) -> str:
        component_meta: Dict[str, Any] = model_metadata.for_component(cls.name)
        file_path: str = component_meta['file_path']
        return cls.name + file_path

    def provide_context(self) -> Dict[str, Any]:
        return {'lookup_table': self.lookup_table}

    def persist(self, model_dir: str) -> Dict[str, Any]:
        lookup_table_path = os.path.join(model_dir, 'word2vec.bin')
        self.lookup_table.save(lookup_table_path, binary=True)
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
        vocab = self.lookup_table.vocab
        return [vocab[t.text].index for t in tokens]


if __name__ == '__main__':
    pass
