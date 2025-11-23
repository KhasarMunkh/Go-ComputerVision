import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#normalizing the pixel values (no reshaping needed for feedforward)
X_train=X_train/255
X_test=X_test/255

#defining feedforward model
FeedForwardModel=Sequential()
    
#flatten input layer (28x28 -> 784)
FeedForwardModel.add(Flatten(input_shape=(28,28)))

#hidden layer 1
FeedForwardModel.add(Dense(512,activation='relu'))
FeedForwardModel.add(Dropout(0.2))

#hidden layer 2
FeedForwardModel.add(Dense(256,activation='relu'))
FeedForwardModel.add(Dropout(0.2))

#hidden layer 3
FeedForwardModel.add(Dense(128,activation='relu'))
FeedForwardModel.add(Dropout(0.2))

#output layer
FeedForwardModel.add(Dense(10,activation='softmax'))

#compiling the model
FeedForwardModel.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#print model summary
print("Feedforward Neural Network Architecture:")
FeedForwardModel.summary()

#fitting the model
print("\nTraining Feedforward Neural Network...")
ff_history = FeedForwardModel.fit(X_train,y_train,epochs=10,validation_split=0.1)

#evaluating the model on test set
ff_test_loss, ff_test_accuracy = FeedForwardModel.evaluate(X_test, y_test)
print(f'\n{"="*50}')
print(f'Feedforward NN Test Accuracy: {ff_test_accuracy:.4f}')
print(f'{"="*50}')
