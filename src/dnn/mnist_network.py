import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load MNIST data
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0

# 2. Define the model
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(10, activation='sigmoid')
])

# 3. Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

# Optionally, save to HDF5
model.save_weights('mnist_dnn.weights.h5')



