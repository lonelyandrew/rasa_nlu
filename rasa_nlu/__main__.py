#!/usr/bin/env python3

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer, Interpreter
from rasa_nlu import config
import logging


training_data = load_data('/home/shixiufeng/Data/trainingdata.txt')
trainer = Trainer(config.load('sample_configs/config_listen_robot.yml'))
trainer.train(training_data)
model_directory = trainer.persist('./projects/default/')
print(model_directory)
interpreter = Interpreter.load(model_directory)
result = interpreter.parse('你好')
print(result)
