import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
import time

print("="*70)
print("TASK 3: Effect of Training Data Size on CNN Performance")
print("="*70)

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#reshaping and normalizing
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)) / 255
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)) / 255

print(f"\nOriginal training data size: {X_train.shape[0]} samples")
print(f"Test data size: {X_test.shape[0]} samples")

def build_cnn():
    """Build the CNN model with 4 conv layers, 4 pooling layers, 1 FC layer"""
    model = Sequential()

    #adding convolution layer 1
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
    #adding pooling layer 1
    model.add(MaxPool2D(2,2))

    #adding convolution layer 2
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    #adding pooling layer 2
    model.add(MaxPool2D(2,2))

    #adding convolution layer 3
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    #adding pooling layer 3
    model.add(MaxPool2D(2,2))

    #adding convolution layer 4
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    #adding pooling layer 4
    model.add(MaxPool2D(2,2))

    #adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))

    #adding output layer
    model.add(Dense(10,activation='softmax'))

    #compiling the model
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

def train_and_evaluate(X_train_subset, y_train_subset, X_test, y_test, data_percentage):
    """Train CNN on subset of data and evaluate"""
    print(f"\n{'='*70}")
    print(f"Training CNN with {data_percentage}% of training data ({len(X_train_subset)} samples)")
    print(f"{'='*70}")

    model = build_cnn()

    start_time = time.time()
    history = model.fit(X_train_subset, y_train_subset,
                       epochs=10,
                       validation_split=0.1,
                       verbose=1)
    train_time = time.time() - start_time

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return test_accuracy, test_loss, train_time, history

# Store results
results = {}

# 1. Train with 100% of training data
print("\n" + "="*70)
print("EXPERIMENT 1: 100% of Training Data (Baseline)")
print("="*70)
acc_100, loss_100, time_100, hist_100 = train_and_evaluate(X_train, y_train, X_test, y_test, 100)
results['100%'] = {'accuracy': acc_100, 'loss': loss_100, 'time': time_100}

# 2. Train with 50% of training data
print("\n" + "="*70)
print("EXPERIMENT 2: 50% of Training Data")
print("="*70)
half_size = len(X_train) // 2
# Use random sampling to get 50%
indices_50 = np.random.choice(len(X_train), half_size, replace=False)
X_train_50 = X_train[indices_50]
y_train_50 = y_train[indices_50]
acc_50, loss_50, time_50, hist_50 = train_and_evaluate(X_train_50, y_train_50, X_test, y_test, 50)
results['50%'] = {'accuracy': acc_50, 'loss': loss_50, 'time': time_50}

# 3. Train with 5% of training data
print("\n" + "="*70)
print("EXPERIMENT 3: 5% of Training Data")
print("="*70)
five_percent_size = int(len(X_train) * 0.05)
# Use random sampling to get 5%
indices_5 = np.random.choice(len(X_train), five_percent_size, replace=False)
X_train_5 = X_train[indices_5]
y_train_5 = y_train[indices_5]
acc_5, loss_5, time_5, hist_5 = train_and_evaluate(X_train_5, y_train_5, X_test, y_test, 5)
results['5%'] = {'accuracy': acc_5, 'loss': loss_5, 'time': time_5}

# Print comparison results
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print(f"\n{'Training Data Size':<25} {'Accuracy':<15} {'Loss':<15} {'Time (s)':<15}")
print("-"*70)
print(f"{'100% (60,000 samples)':<25} {acc_100:.4f} ({acc_100*100:.2f}%){'':<3} {loss_100:.4f}{'':<9} {time_100:.2f}")
print(f"{'50% (30,000 samples)':<25} {acc_50:.4f} ({acc_50*100:.2f}%){'':<3} {loss_50:.4f}{'':<9} {time_50:.2f}")
print(f"{'5% (3,000 samples)':<25} {acc_5:.4f} ({acc_5*100:.2f}%){'':<3} {loss_5:.4f}{'':<9} {time_5:.2f}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Calculate accuracy drops
drop_50 = (acc_100 - acc_50) * 100
drop_5 = (acc_100 - acc_5) * 100

print(f"\nAccuracy compared to 100% baseline:")
print(f"  - 50% data: {drop_50:.2f}% {'decrease' if drop_50 > 0 else 'increase'}")
print(f"  - 5% data: {drop_5:.2f}% decrease")

print(f"\nKey Observations:")
print(f"  1. With 50% of training data:")
print(f"     - Accuracy: {acc_50*100:.2f}% (baseline: {acc_100*100:.2f}%)")
print(f"     - The model still performs well with half the data")
print(f"     - Training time is roughly halved: {time_50:.2f}s vs {time_100:.2f}s")

print(f"\n  2. With only 5% of training data:")
print(f"     - Accuracy: {acc_5*100:.2f}% (baseline: {acc_100*100:.2f}%)")
print(f"     - Significant accuracy drop of {drop_5:.2f}%")
print(f"     - Model is underfitted due to insufficient training examples")
print(f"     - Much faster training: {time_5:.2f}s vs {time_100:.2f}s")

print(f"\nConclusion:")
print(f"  Reducing training data size impacts accuracy. With 50% data, the model")
print(f"  maintains reasonable performance. However, with only 5% data, the model")
print(f"  lacks sufficient examples to learn robust features, resulting in")
print(f"  significant accuracy degradation. More training data generally leads to")
print(f"  better generalization, though with diminishing returns.")
print("="*70)
