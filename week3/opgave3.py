from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# https://www.tensorflow.org/tutorials/images/classification

from pathlib import Path

data_dir = "./Fundus-data"

train_dataset = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(75, 75),
    shuffle=True,
    seed=1234,
    validation_split=1/6,
    subset="training",
    interpolation="bilinear",
)
validation_dataset = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(75, 75),
    shuffle=True,
    seed=1234,
    validation_split=1/6,
    subset="validation",
    interpolation="bilinear",
)

class_names = train_dataset.class_names
num_classes = len(class_names)

model = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(75, 75, 1)),

        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.summary()


model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

predictions = np.argmax(model.predict(validation_dataset), axis=1)

print(class_names)

y = np.concatenate([y for x, y in validation_dataset], axis=0)

cf = tf.math.confusion_matrix(labels=y, predictions=predictions).numpy()

plt.figure()
plt.matshow(cf)
plt.show()
