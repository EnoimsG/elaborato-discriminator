import tensorflow as tf
import numpy as np
import utils
import discriminator
import os
import six
import logging
from datagenerator import DataGenerator
from train import load_dataset

logger = tf.get_logger()
model_params = discriminator.HParams()
datasets = load_dataset('dataset', model_params)
test_set = datasets[2][0] + datasets[2][1]
labels = {dataset: i for (i, dataset) in enumerate(model_params.data_set)}
test_generator = DataGenerator(
    test_set, labels, batch_size=model_params.batch_size, shuffle=False)
model = discriminator.Discriminator(model_params)
predictions = model.test('models/final.model', test_generator)