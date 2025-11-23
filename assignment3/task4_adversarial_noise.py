import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("TASK 4: Adversarial Noise Testing on CNN")
print("="*70)

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#reshaping and normalizing
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255

def build_cnn():
    """Build the CNN model with 4 conv layers, 4 pooling layers, 1 FC layer"""
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# Train the CNN model
print("\nTraining CNN on full training data...")
model = build_cnn()
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Evaluate and find correctly classified images
print("\nEvaluating model on test set...")
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
correct_mask = (predicted_classes == y_test)

print(f"Test accuracy: {np.mean(correct_mask):.4f}")
print(f"Correctly classified: {np.sum(correct_mask)}/{len(y_test)} images")

# Select 2 correctly classified images (one from digit 3 and one from digit 7 for variety)
correctly_classified_indices = np.where(correct_mask)[0]
digit_3_indices = [i for i in correctly_classified_indices if y_test[i] == 3]
digit_7_indices = [i for i in correctly_classified_indices if y_test[i] == 7]

selected_indices = [digit_3_indices[0], digit_7_indices[0]]
selected_images = X_test[selected_indices]
selected_labels = y_test[selected_indices]

print(f"\nSelected images for noise testing:")
print(f"  Image 1: Index {selected_indices[0]}, True Label: {selected_labels[0]}")
print(f"  Image 2: Index {selected_indices[1]}, True Label: {selected_labels[1]}")

# Noise functions
def add_gaussian_noise(image, mean=0, std=0.1):
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_salt_pepper_noise(image, amount=0.05):
    """Add salt and pepper noise to image"""
    noisy_image = image.copy()

    # Salt (white pixels)
    num_salt = int(amount * image.size / 2)
    coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[coords_salt[0], coords_salt[1], coords_salt[2]] = 1

    # Pepper (black pixels)
    num_pepper = int(amount * image.size / 2)
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[coords_pepper[0], coords_pepper[1], coords_pepper[2]] = 0

    return noisy_image

def add_uniform_noise(image, low=-0.1, high=0.1):
    """Add uniform noise to image"""
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_speckle_noise(image, std=0.1):
    """Add speckle (multiplicative) noise to image"""
    noise = np.random.randn(*image.shape)
    noisy_image = image + image * noise * std
    return np.clip(noisy_image, 0, 1)

# Test different noise parameters
noise_tests = []

# Gaussian noise - varying standard deviations
for std in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    noise_tests.append(('Gaussian', {'std': std}, lambda img, s=std: add_gaussian_noise(img, std=s)))

# Salt and pepper noise - varying amounts
for amount in [0.05, 0.1, 0.2, 0.3, 0.5]:
    noise_tests.append(('Salt & Pepper', {'amount': amount}, lambda img, a=amount: add_salt_pepper_noise(img, amount=a)))

# Uniform noise - varying ranges
for high in [0.1, 0.2, 0.3, 0.5]:
    noise_tests.append(('Uniform', {'low': -high, 'high': high}, lambda img, h=high: add_uniform_noise(img, low=-h, high=h)))

# Speckle noise - varying standard deviations
for std in [0.1, 0.2, 0.5, 1.0]:
    noise_tests.append(('Speckle', {'std': std}, lambda img, s=std: add_speckle_noise(img, std=s)))

print("\n" + "="*70)
print("TESTING NOISE PARAMETERS")
print("="*70)

results = []
fooled_examples = []

for i, (img, true_label, img_idx) in enumerate(zip(selected_images, selected_labels, selected_indices)):
    print(f"\n--- Testing Image {i+1} (Original Label: {true_label}) ---")

    # Get original prediction
    orig_pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
    orig_class = np.argmax(orig_pred)
    orig_confidence = np.max(orig_pred)

    print(f"Original prediction: {orig_class} (confidence: {orig_confidence:.4f})")

    for noise_type, params, noise_func in noise_tests:
        noisy_img = noise_func(img)
        noisy_pred = model.predict(noisy_img.reshape(1, 28, 28, 1), verbose=0)
        noisy_class = np.argmax(noisy_pred)
        noisy_confidence = np.max(noisy_pred)

        fooled = (noisy_class != true_label)

        result = {
            'image_idx': img_idx,
            'image_num': i+1,
            'true_label': true_label,
            'noise_type': noise_type,
            'params': params,
            'original_pred': orig_class,
            'original_conf': orig_confidence,
            'noisy_pred': noisy_class,
            'noisy_conf': noisy_confidence,
            'fooled': fooled
        }
        results.append(result)

        if fooled:
            fooled_examples.append({
                'result': result,
                'original_img': img,
                'noisy_img': noisy_img
            })
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            print(f"  ✓ FOOLED! {noise_type} ({param_str}): Predicted {noisy_class} (conf: {noisy_confidence:.4f})")

# Print summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

total_tests = len(results)
fooled_count = sum(1 for r in results if r['fooled'])

print(f"\nTotal noise tests performed: {total_tests}")
print(f"Successfully fooled the CNN: {fooled_count} times ({fooled_count/total_tests*100:.1f}%)")
print(f"Resistant to noise: {total_tests - fooled_count} times ({(total_tests-fooled_count)/total_tests*100:.1f}%)")

if fooled_count > 0:
    print(f"\n✓ YES, we were able to fool the CNN!")
    print(f"\nNoise types that fooled the model:")

    fooled_by_type = {}
    for r in results:
        if r['fooled']:
            noise_key = r['noise_type']
            if noise_key not in fooled_by_type:
                fooled_by_type[noise_key] = []
            fooled_by_type[noise_key].append(r)

    for noise_type, examples in fooled_by_type.items():
        print(f"\n  {noise_type}: {len(examples)} times")
        for ex in examples[:3]:  # Show first 3 examples
            param_str = ', '.join([f"{k}={v}" for k, v in ex['params'].items()])
            print(f"    - {param_str}: {ex['true_label']} → {ex['noisy_pred']} (conf: {ex['noisy_conf']:.4f})")
else:
    print(f"\n✗ The CNN was NOT fooled by any of the tested noise parameters.")

# Visualize results
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

if fooled_count > 0:
    # Show examples of fooled images
    num_examples = min(6, len(fooled_examples))
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_examples):
        example = fooled_examples[idx]
        r = example['result']

        # Original image
        axes[idx, 0].imshow(example['original_img'].reshape(28, 28), cmap='gray')
        axes[idx, 0].set_title(f"Original\nLabel: {r['true_label']}\nPred: {r['original_pred']}")
        axes[idx, 0].axis('off')

        # Noisy image
        axes[idx, 1].imshow(example['noisy_img'].reshape(28, 28), cmap='gray')
        param_str = ', '.join([f"{k}={v}" for k, v in r['params'].items()])
        axes[idx, 1].set_title(f"Noisy ({r['noise_type']})\n{param_str}")
        axes[idx, 1].axis('off')

        # Difference
        diff = np.abs(example['original_img'] - example['noisy_img'])
        axes[idx, 2].imshow(diff.reshape(28, 28), cmap='hot')
        axes[idx, 2].set_title(f"Fooled!\nPredicted: {r['noisy_pred']}\nConf: {r['noisy_conf']:.2f}")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('task4_fooled_examples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: task4_fooled_examples.png")
else:
    # Show examples of noisy images that didn't fool the model
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(2):
        img = selected_images[i]
        true_label = selected_labels[i]

        # Original
        axes[i, 0].imshow(img.reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f"Original\nLabel: {true_label}")
        axes[i, 0].axis('off')

        # Different noise types
        noisy_examples = [
            add_gaussian_noise(img, std=0.3),
            add_salt_pepper_noise(img, amount=0.2),
            add_uniform_noise(img, low=-0.2, high=0.2),
            add_speckle_noise(img, std=0.3)
        ]
        titles = [
            'Gaussian\n(std=0.3)',
            'Salt&Pepper\n(amount=0.2)',
            'Uniform\n(±0.2)',
            'Speckle\n(std=0.3)'
        ]

        for j, (noisy, title) in enumerate(zip(noisy_examples, titles)):
            pred = model.predict(noisy.reshape(1, 28, 28, 1), verbose=0)
            pred_class = np.argmax(pred)

            axes[i, j+1].imshow(noisy.reshape(28, 28), cmap='gray')
            axes[i, j+1].set_title(f"{title}\nPred: {pred_class}")
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.savefig('task4_noise_samples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: task4_noise_samples.png")

# Report tested parameters
print("\n" + "="*70)
print("NOISE PARAMETERS TESTED")
print("="*70)

print("\n1. Gaussian Noise:")
print("   - Standard deviations: 0.1, 0.2, 0.3, 0.5, 0.7, 1.0")
print("   - Mean: 0 (zero-mean)")

print("\n2. Salt & Pepper Noise:")
print("   - Noise amounts: 0.05, 0.1, 0.2, 0.3, 0.5")
print("   - (percentage of pixels affected)")

print("\n3. Uniform Noise:")
print("   - Ranges: [-0.1, 0.1], [-0.2, 0.2], [-0.3, 0.3], [-0.5, 0.5]")

print("\n4. Speckle Noise:")
print("   - Standard deviations: 0.1, 0.2, 0.5, 1.0")
print("   - (multiplicative noise)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if fooled_count > 0:
    print(f"\nThe CNN was successfully fooled {fooled_count}/{total_tests} times.")
    print("The model is vulnerable to certain noise types and intensities.")
    print("Higher noise levels (e.g., Gaussian std>0.5, Salt&Pepper>0.3) are more")
    print("likely to fool the classifier, as they significantly distort the input.")
else:
    print("\nThe CNN showed robustness to the tested noise parameters.")
    print("The model maintained correct predictions despite various noise types.")
    print("This suggests good generalization, though more extreme noise might fool it.")

print("="*70)
