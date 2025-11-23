import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

five_percent_size = int(len(X_train) * 0.05)

indices_5 = np.random.choice(X_train.shape[0], five_percent_size, replace=False)
X_train_5 = X_train[indices_5]
y_train_5 = y_train[indices_5]
#reshaping data
X_train_5 = X_train_5.reshape((X_train_5.shape[0], X_train_5.shape[1], X_train_5.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

#checking the shape after reshaping
print(X_train_5.shape)
print(X_test.shape)

#normalizing the pixel values
X_train_5=X_train_5/255
X_test=X_test/255

#defining model
CNN_Model_5=Sequential()

#adding convolution layer 1
CNN_Model_5.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
#adding pooling layer 1
CNN_Model_5.add(MaxPool2D(2,2))

#adding convolution layer 2
CNN_Model_5.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 2
CNN_Model_5.add(MaxPool2D(2,2))

#adding convolution layer 3
CNN_Model_5.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 3
CNN_Model_5.add(MaxPool2D(2,2))

#adding convolution layer 4
CNN_Model_5.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 4
CNN_Model_5.add(MaxPool2D(2,2))

#adding fully connected layer
CNN_Model_5.add(Flatten())
CNN_Model_5.add(Dense(100,activation='relu'))

#adding output layer
CNN_Model_5.add(Dense(10,activation='softmax'))

#compiling the model
CNN_Model_5.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#print model summary
print("CNN Architecture:")
CNN_Model_5.summary()

#fitting the model
print("\nTraining CNN...")
cnn_history = CNN_Model_5.fit(X_train_5,y_train_5,epochs=10,validation_split=0.1)

#evaluating the model on test set
cnn_5_test_loss, cnn_5_test_accuracy = CNN_Model_5.evaluate(X_test, y_test)
print(f'\n{"="*50}')
print(f'CNN-half-data Test Accuracy: {cnn_5_test_accuracy:.4f}')
print(f'{"="*50}')
print("="*70)
