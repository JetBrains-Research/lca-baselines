 import logging
import h5py
import numpy as np
import tensorflow as tf
from configparser import ConfigParser
from augmentors import Augmentor, BoundaryGrowth, BalanceLabels
from preprocessing import normalize

config = ConfigParser()
config.read('config.ini')

input_size = int(config.get('Data', 'input_size'))
output_size = int(config.get('Data', 'output_size'))
batch_size = int(config.get('Training', 'batch_size'))
num_iterations = int(config.get('Training', 'num_iterations'))

def train(num_iterations):
    raw_intensities = tf.placeholder(tf.float32, shape=(batch_size, input_size, input_size, input_size))
    labelled_objects = tf.placeholder(tf.int32, shape=(batch_size, output_size, output_size, output_size))
    per_voxel_affinities = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size, output_size))
    loss_weights = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size, output_size))

    predicted_affinities = ...
    loss = ...
    gradients = tf.gradients(loss, predicted_affinities)

    with h5py.File('data.hdf5', 'r') as hdf:
        data = hdf['data'][:]

    batch_request = np.random.randint(0, data.shape[0] - batch_size, size=num_iterations)
    snapshot_request = batch_request[::10]

    pipeline = tf.data.TFRecordDataset(filenames=['data.tfrecords']) \
                 .map(lambda x: parse_tfrecord(x)) \
                 .batch(batch_size) \
                 .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
                 .apply(tf.data.experimental.map_and_batch(map_func=lambda x, y: (x, y), batch_size=batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE)) \
                 .map(lambda x, y: (normalize(x), y)) \
                 .map(lambda x, y: (x, BoundaryGrowth()(y))) \
                 .map(lambda x, y: (x, BalanceLabels()(y))) \
                 .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
                 .cache() \
                 .repeat()

    iterator = pipeline.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        print('Starting training...')

        for i in range(num_iterations):
            batch_raw, batch_labels = sess.run(next_element)

            if i in snapshot_request:
                with h5py.File('snapshots/snapshot_{}.hdf5'.format(i), 'w') as hdf:
                    hdf.create_dataset('batch_raw', data=batch_raw)
                    hdf.create_dataset('batch_labels', data=batch_labels[0])

            _, loss_val, gradients_val = sess.run([train_op, loss, gradients], feed_dict={raw_intensities: batch_raw, labelled_objects: batch_labels[0], per_voxel_affinities: batch_labels[1], loss_weights: batch_labels[2]})

            if (i + 1) % 100 == 0:
                print('Iteration {}: loss = {}'.format(i + 1, loss_val))

        print('Training complete.')

if __name__ == '__main__':
    train(num_iterations)