import keras
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, Dense, Flatten, Dropout
from keras.layers.recurrent import RNN, LSTM, LSTMCell
import tensorflow as tf
import os

def copy_hparams(hparams):
	"""Return a copy of an HParams instance."""
	return hparams.values()


class HParams(object):
	def __init__(self):
		self.data_set = ['sketchrnn_tractor.npz', 'sketchrnn_cat.npz']
		self.epochs = 150
		self.max_seq_len = 250
		self.enc_rnn_size = 256  # Size of encoder.
		self.batch_size = 100
		self.recurrent_dropout_prob = 0.90
		self.random_scale_factor = 0.15
		self.augment_stroke_prob = 0.10
		self.is_training = True 

		# Utils
		self.gpu_id = "0"


class Discriminator(object):
	def __init__(self, hps, gpu_mode=True, reuse=False):
		self.hps = hps
		os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
		if not gpu_mode:
			with tf.device('/cpu:0'):
				self.model = self.build_model()
		else:
			with tf.device('/device:gpu:0'):
				tf.compat.v1.logging.info('Model using gpu.')
				self.model = self.build_model()


	def save(self):
		self.model.save(filepath='final.model')

	def build_model(self):
		print('Creating model...')
		model = keras.Sequential()
		forward_layer = LSTM(self.hps.enc_rnn_size, return_sequences=True, recurrent_dropout=0.9)
		backward_layer = LSTM(self.hps.enc_rnn_size,return_sequences=True, recurrent_dropout=0.9, go_backwards=True)
		model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(self.hps.max_seq_len + 1, 5)))
		model.add(Flatten())
		model.add(Dropout(0.2))
		model.add(Dense(64))
		model.add(Dense(1, activation='sigmoid'))
		print('Compiling...')
		model.compile(loss='binary_crossentropy', optimizer='adam')
		model.summary()
		return model

	def train(self, train_generator, valid_generator):
		with tf.device('/device:gpu:0'):
			print('starting training')
			tensorboard = keras.callbacks.TensorBoard(log_dir='logs')
			checkpoint = tf.keras.callbacks.ModelCheckpoint(
				filepath='checkpoints')
			self.model.fit_generator(
				generator=train_generator,
				validation_data=valid_generator,
				epochs=self.hps.epochs,
				callbacks=[tensorboard, checkpoint]
			)

	def test(self, model_path, test_generator):
		print('evaluating...')
		self.model = keras.models.load_model(model_path)
		return self.model.predict(test_generator)

	def load_model(self, model_path):
		self.model = keras.models.load_model(model_path)

	def predict(self, generator):
		return self.model.predict(generator)