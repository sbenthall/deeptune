from __future__ import absolute_import, division, print_function, unicode_literals

import fragment
import numpy as np
import os
import tensorflow as tf
import utils

OUTPUT_DIR = "data/processed"

def preprocess_fragment(fragment):
    np_data = fragment.np_data

    ds = np_data.shape
    tf_input = np_data.reshape(1, ds[0], ds[1], 1)
    #tf_input = tf_input.astype('float32')
    tf_input/=100

    return tf_input

def postprocess_fragment(tf_output):
    np_data = tf_output.numpy() * 100
    ds = np_data.shape
    np_data = np_data.reshape(ds[1], ds[2])

    return np_data


def prepare_tensorflow_datasets():

    frags = list(fragment.from_directory())
    
    ## Getting and stacking the data
    data = np.stack([preprocess_fragment(f)
                     for f
                     in frags],
                    0)
    ds = data.shape
    data = data.reshape(ds[0], ds[2], ds[3], ds[4])
    
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
