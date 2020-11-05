from tensorflow import keras

# magische data invoeren
# magie train_images, train_labels    6 deel = train  1 deel test 

model = keras.Sequential(
    [
        keras.layers.Dense(75*75, activation="relu", input_shape=(75*75,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(39, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(train_images, train_labels, epochs=6)
