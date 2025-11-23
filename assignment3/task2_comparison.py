import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import time

print("="*70)
print("TASK 2: CNN vs Feedforward Neural Network Comparison")
print("="*70)

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#prepare data for CNN
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)) / 255
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)) / 255

#prepare data for feedforward (no reshape needed)
X_train_ff = X_train / 255
X_test_ff = X_test / 255

print("\n" + "="*70)
print("Building and Training CNN Model")
print("="*70)

#build CNN
cnn_model = Sequential()
cnn_model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
cnn_model.add(MaxPool2D(2,2))
cnn_model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
cnn_model.add(MaxPool2D(2,2))
cnn_model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
cnn_model.add(MaxPool2D(2,2))
cnn_model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
cnn_model.add(MaxPool2D(2,2))
cnn_model.add(Flatten())
cnn_model.add(Dense(100,activation='relu'))
cnn_model.add(Dense(10,activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print("\nCNN Architecture:")
cnn_model.summary()

start_time = time.time()
cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=10, validation_split=0.1, verbose=1)
cnn_train_time = time.time() - start_time

cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)

print("\n" + "="*70)
print("Building and Training Feedforward Neural Network Model")
print("="*70)

#build feedforward network
ff_model = Sequential()
ff_model.add(Flatten(input_shape=(28,28)))
ff_model.add(Dense(512,activation='relu'))
ff_model.add(Dropout(0.2))
ff_model.add(Dense(256,activation='relu'))
ff_model.add(Dropout(0.2))
ff_model.add(Dense(128,activation='relu'))
ff_model.add(Dropout(0.2))
ff_model.add(Dense(10,activation='softmax'))

ff_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print("\nFeedforward NN Architecture:")
ff_model.summary()

start_time = time.time()
ff_history = ff_model.fit(X_train_ff, y_train, epochs=10, validation_split=0.1, verbose=1)
ff_train_time = time.time() - start_time

ff_loss, ff_accuracy = ff_model.evaluate(X_test_ff, y_test, verbose=0)

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print(f"\nCNN Model:")
print(f"  - Test Accuracy: {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
print(f"  - Test Loss: {cnn_loss:.4f}")
print(f"  - Training Time: {cnn_train_time:.2f} seconds")
print(f"  - Total Parameters: {cnn_model.count_params():,}")

print(f"\nFeedforward NN Model:")
print(f"  - Test Accuracy: {ff_accuracy:.4f} ({ff_accuracy*100:.2f}%)")
print(f"  - Test Loss: {ff_loss:.4f}")
print(f"  - Training Time: {ff_train_time:.2f} seconds")
print(f"  - Total Parameters: {ff_model.count_params():,}")

print(f"\nPerformance Difference:")
accuracy_diff = (cnn_accuracy - ff_accuracy) * 100
if cnn_accuracy > ff_accuracy:
    print(f"  - CNN is {accuracy_diff:.2f}% more accurate than Feedforward NN")
else:
    print(f"  - Feedforward NN is {-accuracy_diff:.2f}% more accurate than CNN")

print(f"\nConclusion:")
print(f"  The {'CNN' if cnn_accuracy > ff_accuracy else 'Feedforward NN'} performs better on MNIST dataset.")
print(f"  CNNs are typically better for image tasks due to spatial feature extraction,")
print(f"  while feedforward networks treat pixels independently.")
print("="*70)
