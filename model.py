import tensorflow as tf
import load_data as ld

(train_dataset, test_dataset) = ld.shuffle(
    *ld.prepare_tensorflow_datasets())

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
