import tensorflow as tf
import numpy as np
import utils
import discriminator
import logging
import os
from datagenerator import DataGenerator
from train import load_dataset
from sklearn.metrics import accuracy_score, precision_score

model_params = discriminator.HParams()
model = discriminator.Discriminator(model_params)
model.load_model('models/final.model')

cat_dir = 'experiments_results/cat/'
tractor_dir = 'experiments_results/tractor/'

def transform_draws(draws):
    max_seq_len = 137
    test_set = utils.DataLoader(
            draws,
            100,
            max_seq_length=max_seq_len,
            random_scale_factor=0.15,
            augment_stroke_prob=0.1)

    normalizing_scale_factor = test_set.calculate_normalizing_scale_factor()
    test_set.normalize(normalizing_scale_factor)
    test_set.strokes = [utils.to_big_strokes(
        stroke, max_seq_len) for stroke in test_set.strokes]
    test_set.strokes = [
        np.insert(stroke, 0, [0, 0, 1, 0, 0], axis=0) for stroke in test_set.strokes]
    return test_set.strokes


for filename in os.listdir(cat_dir):
    draws = np.load(os.path.join(cat_dir, filename), allow_pickle=True)
    topredict = transform_draws(draws)
    predictions = model.predict(np.array(topredict))
    true_y = np.ones(1000)
    pred_y = np.rint(predictions)
    print('Accuracy exp cat_', filename, ' : ', accuracy_score(true_y, pred_y))


for filename in os.listdir(tractor_dir):
    draws = np.load(os.path.join(tractor_dir, filename), allow_pickle=True)
    topredict = transform_draws(draws)
    predictions = model.predict(np.array(topredict))
    true_y = np.zeros(1000)
    pred_y = np.rint(predictions)
    print('Accuracy exp tractor_', filename, ' : ', accuracy_score(true_y, pred_y))


draws = np.load(os.path.join('experiments_results', 'their_cat.npy'), allow_pickle=True)
topredict = transform_draws(draws)
predictions = model.predict(np.array(topredict))
true_y = np.ones(1000)
pred_y = np.rint(predictions)
print('Accuracy exp their_cat: ', accuracy_score(true_y, pred_y))

draws = np.load(os.path.join('experiments_results', 'their_tractor.npy'), allow_pickle=True)
topredict = transform_draws(draws)
predictions = model.predict(np.array(topredict))
true_y = np.zeros(1000)
pred_y = np.rint(predictions)
print('Accuracy exp their_tractor: ', accuracy_score(true_y, pred_y))
