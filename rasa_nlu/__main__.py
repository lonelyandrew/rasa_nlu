#!/usr/bin/env python3

from typing import Dict, Any
import logging
import argparse
from datetime import datetime

from rasa_nlu.training_data import load_data, TrainingData
from rasa_nlu.model import Trainer, Interpreter
from rasa_nlu import config
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.evaluate import run_evaluation, return_entity_results, \
    return_results, run_cv_evaluation


logger = logging.getLogger(__name__)


def argparse_config():
    parser = argparse.ArgumentParser(description='Test Rasa NLU.')
    parser.add_argument('--mode', default='parse', const='parse',
                        choices=['train', 'parse', 'evaluate'], nargs='?')
    parser.add_argument('--model', nargs='?', default='')
    args = parser.parse_args()
    return args


def logging_config():
    log_file_name = f'log/{args.mode}-{datetime.now()}.log'
    log_format_str = '[%(asctime)s]-[%(name)s]-[%(levelname)s]-%(message)s'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG,
                        filemode='w+', format=log_format_str)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def train(training_data_path: str, config_path: str, save_dir: str) -> str:
    training_data: TrainingData = load_data(training_data_path)
    trainer_config: RasaNLUModelConfig = config.load(config_path)
    trainer = Trainer(trainer_config)
    trainer.train(training_data)
    return trainer.persist(save_dir)


def parse(model_dir: str, text: str) -> Dict[str, Any]:
    interpreter = Interpreter.load(model_dir)
    return interpreter.parse(text)


def evaluate(model_dir: str, test_data_path: str, config_path: str=None,
             mode: str='evaluation', folds: int=5):
    logger.info(f'MODEL DIR: {model_dir}')
    if mode == 'evaluation':
        run_evaluation(test_data_path, model_dir, errors_filename=None)
    elif mode == 'cross_validation':
        model_config = config.load(config_path)
        data = load_data(test_data_path)
        results, entity_results = run_cv_evaluation(data, folds, model_config)
        if any(results):
            logger.info("Intent evaluation results")
            return_results(results.train, "train")
            return_results(results.test, "test")
        if any(entity_results):
            logger.info("Entity evaluation results")
            return_entity_results(entity_results.train, "train")
            return_entity_results(entity_results.test, "test")
    else:
        raise ValueError(f'Invalid Mode "{mode}". Please set the mode '
                         'either "cross_validation" or "evaluation".')


if __name__ == '__main__':
    args = argparse_config()
    logging_config()
    debug_mode = 'debug' if __debug__ else 'product'
    logger.info(f'MODE: {args.mode} ({debug_mode})')
    training_data_path = '/home/shixiufeng/Data/corpus_intent/training_data.json'  # noqa
    config_path = 'sample_configs/config_listen_robot.yml'
    save_dir = './projects/default/'
    test_data_path = '/home/shixiufeng/Data/corpus_intent/test_intent.json'

    if args.mode in ['train', 'parse']:
        model_dir = train(training_data_path, config_path, save_dir)
        logging.info('MODEL DIR: ' + model_dir)

    if args.mode == 'parse':
        text = '有点贵'
        result = parse(model_dir, text)
        logging.info(result)

    if args.mode == 'evaluate':
        evaluate(args.model, test_data_path)
