from tensorflow import keras
import tensorflow as tf

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

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

class_names = train_dataset.class_names
print(class_names)
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
