from __future__ import absolute_import, division, print_function, unicode_literals
 
import numpy as np
import os
import tensorflow as tf
import utils

OUTPUT_DIR = "data/processed"

def prepare_tensorflow_datasets():

    filenames = utils.onlyfiles(OUTPUT_DIR)

    ## Getting the labels
    labels = [f.split("---")[0] for f in filenames]
    unique_labels = list(set(labels))
    num_labels = np.array([unique_labels.index(label)
                           for label
                           in labels])


    ## Getting and stacking the data
    data = np.stack([np.load(os.path.join(OUTPUT_DIR,f),
                             allow_pickle=True)
                     for f
                     in filenames],
                    0)

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

    return train_dataset, test_dataset


def shuffle(train_dataset, test_dataset):
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 50

    train_dataset = train_dataset.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset
