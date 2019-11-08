from __future__ import absolute_import, division, print_function, unicode_literals

import fragment
import numpy as np
import os
import tensorflow as tf
import utils

OUTPUT_DIR = "data/processed"

def prepare_tensorflow_datasets():

    frags = list(fragment.from_directory())
    
    ## Getting and stacking the data
    data = np.stack([f.np_data
                     for f
                     in frags],
                    0)

    ds = data.shape

    print(data.shape)
    data = data.reshape(ds[0], ds[1], ds[2], 1)
    data = data.astype('float32')
    data/=100


    ## Getting the labels
    labels = [f.song for f in frags]
    unique_labels = list(set(labels))
    num_labels = np.array([unique_labels.index(label)
                           for label
                           in labels])

    

    ## shuffle the data
    p = np.random.permutation(len(num_labels))
    num_labels = num_labels[p]
    data = data[p,:,:]

    # 30% of data is for testing.
    train_count = int(len(labels) * 7 / 10)

    train_examples = data[:train_count]
    train_labels = num_labels[:train_count]
    test_examples = data[:train_count]
    test_labels = num_labels[:train_count]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_examples, test_labels))

    return train_dataset, test_dataset, max(num_labels) + 1


def shuffle(train_dataset, test_dataset):
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 50

    train_dataset = train_dataset.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset
