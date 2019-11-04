import load_data as ld
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

### where is this coming from?
input_shape = (1025,44,1)

(train_dataset, test_dataset, num_categories) = ld.prepare_tensorflow_datasets()
(train_dataset, test_dataset) = ld.shuffle(train_dataset, test_dataset)

model = tf.keras.Sequential([
    Conv2D(32, (3, 3),
           activation='relu',
           padding='same',
           input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3),
           activation='relu',
           padding='same',
           input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(.1),
    Dense(num_categories, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()

history = model.fit(train_dataset,
                    epochs=4
)

plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

## testing
print("Testing:")

test_loss, test_acc = model.evaluate(test_dataset,
                                     verbose=2)

print(test_loss, test_acc)

model.save("song_model.h5")
