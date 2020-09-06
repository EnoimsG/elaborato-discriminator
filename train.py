import tensorflow as tf
import numpy as np
import utils
import discriminator
import os
import six
import logging
from datagenerator import DataGenerator

logger = tf.get_logger()


def console_entry_point():
    main()


def main():
    """Load model params, save config file and start trainer."""
    model_params = discriminator.HParams()
    trainer(model_params)


def trainer(model_params):
    datasets = load_dataset('dataset', model_params)
    train_set = datasets[0][0] + datasets[0][1]
    valid_set = datasets[1][0] + datasets[1][1]
    test_set = datasets[2][0] + datasets[2][1]
    labels = {dataset: i for (i, dataset) in enumerate(model_params.data_set)}
    train_generator = DataGenerator(
        train_set, labels, batch_size=model_params.batch_size, shuffle=True)
    valid_generator = DataGenerator(
        valid_set, labels, batch_size=model_params.batch_size, shuffle=True)
    model = discriminator.Discriminator(model_params)
    model.train(train_generator, valid_generator)
    model.save()
    print('Done!')


def load_dataset(data_dir, model_params, inference_mode=False):
    """Loads the .npz file, and splits the set into train/valid/test."""

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    train_data = []
    valid_data = []
    test_data = []

    dataset_lengths = []

    all_strokes = []

    for i, dataset in enumerate(datasets):
        data_filepath = os.path.join(data_dir, dataset)
        if six.PY3:
            tmp_data = np.load(
                data_filepath, encoding='latin1', allow_pickle=True)
        else:
            tmp_data = np.load(data_filepath, allow_pickle=True)

        all_strokes = np.concatenate(
            (all_strokes, tmp_data['train'], tmp_data['valid'], tmp_data['test']))

    max_seq_len = utils.get_max_len(all_strokes)
    model_params.max_seq_len = max_seq_len
    print('Max sequence length: ', max_seq_len)

    for i, dataset in enumerate(datasets):
        data_filepath = os.path.join(data_dir, dataset)
        if six.PY3:
            data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        else:
            data = np.load(data_filepath, allow_pickle=True)
        logger.info('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']), dataset))
        train_strokes = data['train']
        valid_strokes = data['valid']
        test_strokes = data['test']

        train_set = utils.DataLoader(
            train_strokes,
            model_params.batch_size,
            max_seq_length=max_seq_len,
            random_scale_factor=model_params.random_scale_factor,
            augment_stroke_prob=model_params.augment_stroke_prob)

        normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
        train_set.normalize(normalizing_scale_factor)
        train_set.strokes = [utils.to_big_strokes(
            stroke, max_seq_len) for stroke in train_set.strokes]
        train_set.strokes = [
            np.insert(stroke, 0, [0, 0, 1, 0, 0], axis=0) for stroke in train_set.strokes]

        valid_set = utils.DataLoader(
            valid_strokes,
            model_params.batch_size,
            max_seq_length=max_seq_len,
            random_scale_factor=model_params.random_scale_factor,
            augment_stroke_prob=model_params.augment_stroke_prob)

        valid_set.normalize(normalizing_scale_factor)
        valid_set.strokes = [utils.to_big_strokes(
            stroke, max_seq_len) for stroke in valid_set.strokes]
        valid_set.strokes = [
            np.insert(stroke, 0, [0, 0, 1, 0, 0], axis=0) for stroke in valid_set.strokes]

        test_set = utils.DataLoader(
            test_strokes,
            model_params.batch_size,
            max_seq_length=max_seq_len,
            random_scale_factor=model_params.random_scale_factor,
            augment_stroke_prob=model_params.augment_stroke_prob)

        test_set.normalize(normalizing_scale_factor)
        test_set.strokes = [utils.to_big_strokes(
            stroke, max_seq_len) for stroke in test_set.strokes]
        test_set.strokes = [
            np.insert(stroke, 0, [0, 0, 1, 0, 0], axis=0) for stroke in test_set.strokes]

        train_sketches = [{'dataset': dataset, 'draw': sketch}
                           for sketch in train_set.strokes]
        valid_sketches = [{'dataset': dataset, 'draw': sketch}
                           for sketch in valid_set.strokes]
        test_sketches = [{'dataset': dataset, 'draw': sketch}
                          for sketch in test_set.strokes]

        train_data.append(train_sketches)
        valid_data.append(valid_sketches)
        test_data.append(test_sketches)

    return [train_data, valid_data, test_data]


if __name__ == '__main__':
    console_entry_point()
