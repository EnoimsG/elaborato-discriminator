import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, batch_size=32, shuffle=True):
        'Initialization'
        # data = [{'dataset': ..., 'strokes': ....}, ...]
        # labels = {'cat' : 0 , 'tractor' :  1}
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.data = data
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_batch = [self.data[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(data_batch)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print('update indexes')
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_batch):
        'Generates data containing batch_size samples'
        X = [data['draw'] for data in data_batch]
        Y = [self.labels[data['dataset']] for data in data_batch]

        return np.array(X), np.array(Y)
