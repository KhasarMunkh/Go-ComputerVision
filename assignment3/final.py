import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshaping and normalizing
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255

# checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)

# normalizing the pixel values
X_train = X_train / 255
X_test = X_test / 255


def build_cnn():
    """Build the CNN model with 4 conv layers, 4 pooling layers, 1 FC layer"""
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1))
    )
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


model = build_cnn()
model.fit(X_train, y_train, epochs=10, validation_split=0.1)
# evaluating the model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")


# Implement a feedforward NN model for comparison
def build_ffnn():
    """Build a Feedforward NN model with 2 hidden layers"""
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="softmax"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# defining feedforward model
FeedForwardModel = Sequential()

# flatten input layer (28x28 -> 784)
FeedForwardModel.add(Flatten(input_shape=(28, 28)))

# hidden layer 1
FeedForwardModel.add(Dense(512, activation="relu"))
FeedForwardModel.add(Dropout(0.2))

# hidden layer 2
FeedForwardModel.add(Dense(256, activation="relu"))
FeedForwardModel.add(Dropout(0.2))

# hidden layer 3
FeedForwardModel.add(Dense(128, activation="relu"))
FeedForwardModel.add(Dropout(0.2))

# output layer
FeedForwardModel.add(Dense(10, activation="softmax"))

# compiling the model
FeedForwardModel.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# print model summary
print("Feedforward Neural Network Architecture:")
FeedForwardModel.summary()

# fitting the model
print("\nTraining Feedforward Neural Network...")
ff_history = FeedForwardModel.fit(X_train, y_train, epochs=10, validation_split=0.1)

# evaluating the model on test set
ff_test_loss, ff_test_accuracy = FeedForwardModel.evaluate(X_test, y_test)
print(f"\n{'=' * 50}")
print(f"Feedforward NN Test Accuracy: {ff_test_accuracy:.4f}")
print(f"{'=' * 50}")
