#!/usr/bin/env python3

'''Jieba Tokenizer.
'''

import glob
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import jieba

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Message, Metadata
from rasa_nlu.tokenizers import Token, Tokenizer
from rasa_nlu.training_data import TrainingData

LOGGER = logging.getLogger(__name__)

JIEBA_USER_DICTIONARY_DIR = 'tokenizer_jieba'


class JiebaTokenizer(Tokenizer, Component):
    '''Jieba Tokenizer.
    '''
    name: str = 'tokenizer_jieba'

    provides: List[str] = ['tokens']

    language_list: List[str] = ['zh']

    defaults: Dict[str, Any] = {
        'user_dict_dir': None  # default don't load user dictionary
    }

    def __init__(self, component_config: Dict[str, Any]=None) -> None:
        super(JiebaTokenizer, self).__init__(component_config)
        user_dict_dir: str = self.component_config.get('user_dict_dir')
        self.user_dict_dir: Optional[str] = user_dict_dir

    @classmethod
    def required_packages(cls) -> List[str]:
        return ['jieba']

    @staticmethod
    def load_user_dictionary(user_dict_dir: str) -> None:
        '''Load the dictionaries for the Jieba tokenizer.

        Args:
            user_dict_dir: The dir of the user dicts.
        '''
        user_dict_path_list: List[str] = glob.glob(f'{user_dict_dir}/*')
        for user_dict_path in user_dict_path_list:
            LOGGER.info(f'Loading Jieba User Dictionary at {user_dict_path}.')
            jieba.load_userdict(user_dict_path)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        for example in training_data.training_examples:
            example.set('tokens', self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        tokens = self.tokenize(message.text)
        message.set('tokens', tokens)
        # tokenized_token_text_list = [token.text for token in tokens]
        # message.set('tokenized_text', '-'.join(tokenized_token_text_list),
        #             add_to_output=True)

    def tokenize(self, text: str) -> List[Token]:
        '''Tokenize the sentence.
        '''
        if self.user_dict_dir is not None:
            self.load_user_dictionary(self.user_dict_dir)
        tokenized = jieba.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]
        return tokens

    @classmethod
    def load(cls, model_dir: str, model_metadata: Optional[Metadata]=None,
             cached_component: Optional['JiebaTokenizer']=None,
             **kwargs: Any) -> 'JiebaTokenizer':
        if model_metadata is not None:
            meta = model_metadata.for_component(cls.name)
            relative_user_dict_dir = meta.get('user_dict_dir')
            if relative_user_dict_dir is not None:
                dictionary_path = os.path.join(model_dir,
                                               relative_user_dict_dir)
                meta['user_dict_dir'] = dictionary_path
        else:
            meta = {'user_dict_dir': None}
        return cls(meta)

    @staticmethod
    def copy_files_dir_to_dir(input_dir: str, output_dir: str) -> None:
        '''Copy files from one dir to another.
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob(f'{input_dir}/*')
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self, model_dir: str) -> Dict[str, Any]:
        model_dict_dir = None

        if self.user_dict_dir is not None:
            target_dict_dir = os.path.join(model_dir,
                                           JIEBA_USER_DICTIONARY_DIR)
            self.copy_files_dir_to_dir(self.user_dict_dir, target_dict_dir)
            model_dict_dir = JIEBA_USER_DICTIONARY_DIR
        return {'user_dict_dir': model_dict_dir}
