import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#reshaping data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

#checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)

#normalizing the pixel values
X_train=X_train/255
X_test=X_test/255

#defining model
CNN_Model=Sequential()

#adding convolution layer 1
CNN_Model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
#adding pooling layer 1
CNN_Model.add(MaxPool2D(2,2))

#adding convolution layer 2
CNN_Model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 2
CNN_Model.add(MaxPool2D(2,2))

#adding convolution layer 3
CNN_Model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 3
CNN_Model.add(MaxPool2D(2,2))

#adding convolution layer 4
CNN_Model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#adding pooling layer 4
CNN_Model.add(MaxPool2D(2,2))

#adding fully connected layer
CNN_Model.add(Flatten())
CNN_Model.add(Dense(100,activation='relu'))

#adding output layer
CNN_Model.add(Dense(10,activation='softmax'))

#compiling the model
CNN_Model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#print model summary
print("CNN Architecture:")
CNN_Model.summary()

#fitting the model
print("\nTraining CNN...")
cnn_history = CNN_Model.fit(X_train,y_train,epochs=10,validation_split=0.1)

#evaluating the model on test set
cnn_test_loss, cnn_test_accuracy = CNN_Model.evaluate(X_test, y_test)
print(f'\n{"="*50}')
print(f'CNN Test Accuracy: {cnn_test_accuracy:.4f}')
print(f'{"="*50}')
