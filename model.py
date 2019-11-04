import tensorflow as tf
import load_data as ld

### where is this coming from?
input_shape = (1025,44,1)

(train_dataset, test_dataset, num_categories) = ld.prepare_tensorflow_datasets()
(train_dataset, test_dataset) = ld.shuffle(train_dataset, test_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (500, 5),
                           activation='relu',
                           input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(num_categories, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()

history = model.fit(train_dataset)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.evaluate(test_dataset)


