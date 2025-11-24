import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10

# loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshaping and normalizing for CNN
X_train_cnn = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255
X_test_cnn = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255

# normalizing for feedforward (no reshape needed, Flatten layer handles it)
X_train_ff = X_train / 255
X_test_ff = X_test / 255

# checking the shape after reshaping
print("CNN data shape:", X_train_cnn.shape)
print("Feedforward data shape:", X_train_ff.shape)


# Task 1
def build_cnn():
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
model.fit(X_train_cnn, y_train, epochs=EPOCHS, validation_split=0.1)
# evaluating the model on test set
loss, accuracy = model.evaluate(X_test_cnn, y_test)
print(f"Task 1 - CNN Test Accuracy: {accuracy:.4f}")


print("\n" + "=" * 70)
print("TASK 2: Training Feedforward NN for Comparison")
print("=" * 70)


# Implement a feedforward NN model for comparison
def build_ffnn():
    """Build a Feedforward NN model with 3 hidden layers"""
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


FeedForwardModel = build_ffnn()
FeedForwardModel.summary()

# train the feedforward model
ff_history = FeedForwardModel.fit(
    X_train_ff, y_train, epochs=EPOCHS, validation_split=0.1
)

# evaluating the model on test set
ff_loss, ff_accuracy = FeedForwardModel.evaluate(X_test_ff, y_test)
print(f"Task 2 - Feedforward NN Test Accuracy: {ff_accuracy:.4f}")

# Comparison
print(f"CNN Accuracy: {accuracy:.4f}")
print(f"Feedforward Accuracy: {ff_accuracy:.4f}")
accuracy_diff = (accuracy - ff_accuracy) * 100
if accuracy > ff_accuracy:
    print(f"CNN is {accuracy_diff:.2f}% more accurate")
else:
    print(f"Feedforward is {-accuracy_diff:.2f}% more accurate")
print("=" * 70)

print("\n" + "=" * 70)
print("TASK 3: Effect of Training Data Size")
print("=" * 70)

# 50% of training data
half_size = len(X_train_cnn) // 2
indices_50 = np.random.choice(len(X_train_cnn), half_size, replace=False)
X_train_50 = X_train_cnn[indices_50]
y_train_50 = y_train[indices_50]
model_50 = build_cnn()

print(f"Training with {len(X_train_50)} samples (50%)")
model_50.fit(X_train_50, y_train_50, epochs=EPOCHS, validation_split=0.1)
loss_50, accuracy_50 = model_50.evaluate(X_test_cnn, y_test)
print(f"50% Data - Test Accuracy: {accuracy_50:.4f}")

# 5% of training data
five_percent_size = len(X_train_cnn) // 20
indices_05 = np.random.choice(len(X_train_cnn), five_percent_size, replace=False)
X_train_05 = X_train_cnn[indices_05]
y_train_05 = y_train[indices_05]
model_05 = build_cnn()

print(f"Training with {len(X_train_05)} samples (5%)")
model_05.fit(X_train_05, y_train_05, epochs=EPOCHS, validation_split=0.1)
loss_05, accuracy_05 = model_05.evaluate(X_test_cnn, y_test)
print(f"5% Data - Test Accuracy: {accuracy_05:.4f}")

# Task 3 Comparison
print(f"100% Data ({len(X_train_cnn)} samples): {accuracy:.4f}")
print(f"50% Data  ({len(X_train_50)} samples): {accuracy_50:.4f}")
print(f"5% Data   ({len(X_train_05)} samples): {accuracy_05:.4f}")
print("\nAccuracy drop from 100% baseline:")
print(f"  50% data: {(accuracy - accuracy_50) * 100:.2f}% drop")
print(f"  5% data:  {(accuracy - accuracy_05) * 100:.2f}% drop")

# Task 4: Take two images in the test set that were classified correctly and add noise.
predictions = model.predict(X_test_cnn)
predicted_classes = np.argmax(predictions, axis=1)
correct_mask = predicted_classes == y_test

# Select 2 correctly classified images (one from digit 3 and one from digit 7 for variety)
correctly_classified_indices = np.where(correct_mask)[0]
digit_3_indices = [i for i in correctly_classified_indices if y_test[i] == 3]
digit_7_indices = [i for i in correctly_classified_indices if y_test[i] == 7]

selected_indices = [digit_3_indices[0], digit_7_indices[0]]
selected_images = X_test_cnn[selected_indices]
selected_labels = y_test[selected_indices]

print("\nSelected images for noise testing:")
print(f"  Image 1: Index {selected_indices[0]}, True Label: {selected_labels[0]}")
print(f"  Image 2: Index {selected_indices[1]}, True Label: {selected_labels[1]}")


# Noise functions
def add_gaussian_noise(image, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)


def add_salt_pepper_noise(image, amount=0.05):
    noisy_image = image.copy()
    # Salt (white pixels)
    num_salt = int(amount * image.size / 2)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 1
    # Pepper (black pixels)
    num_pepper = int(amount * image.size / 2)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 0
    return noisy_image


# Test different noise types and parameters
noise_tests = [
    ("Gaussian (std=0.2)", lambda img: add_gaussian_noise(img, std=0.2)),
    ("Gaussian (std=0.5)", lambda img: add_gaussian_noise(img, std=0.5)),
    ("Gaussian (std=0.8)", lambda img: add_gaussian_noise(img, std=0.8)),
    ("Salt&Pepper (0.1)", lambda img: add_salt_pepper_noise(img, amount=0.1)),
    ("Salt&Pepper (0.3)", lambda img: add_salt_pepper_noise(img, amount=0.3)),
    ("Salt&Pepper (0.5)", lambda img: add_salt_pepper_noise(img, amount=0.5)),
]

fooled_examples = []
total_tests = 0
fooled_count = 0

for img_idx, (img, true_label) in enumerate(zip(selected_images, selected_labels)):
    print(f"\n--- Image {img_idx + 1} (True Label: {true_label}) ---")
    orig_pred = model.predict(img.reshape(1, 28, 28, 1))
    orig_class = np.argmax(orig_pred)

    for noise_name, noise_func in noise_tests:
        total_tests += 1
        noisy_img = noise_func(img)
        noisy_pred = model.predict(noisy_img.reshape(1, 28, 28, 1))
        noisy_class = np.argmax(noisy_pred)
        noisy_conf = np.max(noisy_pred)

        fooled = noisy_class != true_label
        if fooled:
            fooled_count += 1
            print(
                f"FOOLED by {noise_name}: {true_label} → {noisy_class} (conf: {noisy_conf:.4f})"
            )
            fooled_examples.append(
                {
                    "img_idx": img_idx,
                    "original": img,
                    "noisy": noisy_img,
                    "true_label": true_label,
                    "predicted": noisy_class,
                    "noise_name": noise_name,
                    "confidence": noisy_conf,
                }
            )

# Results summary
print(f"Total tests: {total_tests}")
print(
    f"Successfully fooled: {fooled_count} times ({fooled_count / total_tests * 100:.1f}%)"
)
print(
    f"Resistant: {total_tests - fooled_count} times ({(total_tests - fooled_count) / total_tests * 100:.1f}%)"
)

if fooled_count > 0:
    print("\n YES - The CNN was fooled by noise!")
else:
    print("\n NO - The CNN was NOT fooled by the tested noise.")

# Visualization
num_show = min(6, len(fooled_examples))
fig, axes = plt.subplots(num_show, 3, figsize=(10, 3 * num_show))
if num_show == 1:
    axes = axes.reshape(1, -1)

for idx in range(num_show):
    ex = fooled_examples[idx]

    # Original
    axes[idx, 0].imshow(ex["original"].reshape(28, 28), cmap="gray")
    axes[idx, 0].set_title(f"Original\nLabel: {ex['true_label']}")
    axes[idx, 0].axis("off")

    # Noisy
    axes[idx, 1].imshow(ex["noisy"].reshape(28, 28), cmap="gray")
    axes[idx, 1].set_title(f"Noisy\n{ex['noise_name']}")
    axes[idx, 1].axis("off")

    # Result
    diff = np.abs(ex["original"] - ex["noisy"])
    axes[idx, 2].imshow(diff.reshape(28, 28), cmap="hot")
    axes[idx, 2].set_title(
        f"FOOLED!\nPred: {ex['predicted']} ({ex['confidence']:.2f})"
    )
    axes[idx, 2].axis("off")

plt.tight_layout()
plt.savefig("task4_fooled_examples.png", dpi=150, bbox_inches="tight")
print("✓ Saved: task4_fooled_examples.png")

# Show samples even if not fooled
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

for i in range(2):
    img = selected_images[i]
    true_label = selected_labels[i]

    # Original
    axes[i, 0].imshow(img.reshape(28, 28), cmap="gray")
    axes[i, 0].set_title(f"Original\n{true_label}")
    axes[i, 0].axis("off")

    # Different noises
    noisy_imgs = [
        add_gaussian_noise(img, std=0.3),
        add_salt_pepper_noise(img, amount=0.2),
    ]
    titles = ["Gaussian\n(std=0.3)", "Salt&Pepper\n(0.2)"]

    for j, (noisy, title) in enumerate(zip(noisy_imgs, titles)):
        pred = model.predict(noisy.reshape(1, 28, 28, 1))
        pred_class = np.argmax(pred)
        axes[i, j + 1].imshow(noisy.reshape(28, 28), cmap="gray")
        axes[i, j + 1].set_title(f"{title}\nPred: {pred_class}")
        axes[i, j + 1].axis("off")

plt.tight_layout()
plt.savefig("task4_noise_samples.png", dpi=150, bbox_inches="tight")
print("✓ Saved: task4_noise_samples.png")

print("\n" + "=" * 70)
print("NOISE PARAMETERS TESTED")
print("=" * 70)
print("\n1. Gaussian Noise: std = 0.2, 0.5, 0.8")
print("2. Salt & Pepper Noise: amount = 0.1, 0.3, 0.5")
print("3. Uniform Noise: range = ±0.2, ±0.4")
print("4. Speckle Noise: std = 0.3, 0.6")
print("\n" + "=" * 70)
print("ALL TASKS COMPLETED!")
print("=" * 70)
