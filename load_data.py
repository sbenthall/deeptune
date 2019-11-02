from __future__ import absolute_import, division, print_function, unicode_literals
 
import numpy as np
import os
import tensorflow as tf


OUTPUT_DIR = "data/processed"

filenames = [f
             for f
             in os.listdir(OUTPUT_DIR)
             if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

## Getting the labels
labels = [f.split("---")[0] for f in filenames]
unique_labels = list(set(labels))
num_labels = np.array([unique_labels.index(label) for label in labels])


## Getting and stacking the data
data = np.stack([np.load(os.path.join(OUTPUT_DIR,f),
                         allow_pickle=True)
                 for f
                 in filenames],
                0)


p = np.random.permutation(len(num_labels))
print(len(num_labels))
print(len(p))
num_labels = num_labels[p]
data = data[p,:,:]

train_count = int(len(labels) * 2 / 3)

train_examples = data[:train_count]
train_labels = num_labels[:train_count]
test_examples = data[:train_count]
test_labels = num_labels[:train_count]


train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
  
#train_examples = []
#train_labels = []
#test_examples = []
#test_labels = []

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1025, 44)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model.evaluate(test_dataset)
