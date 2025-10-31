import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

building_path = "building.jpg"
cheetah_path = "cheetah.jpg"
elephant_path = "elephant.jpg"
output_dir_part_a = "output_part_a"
output_dir_part_b = "output_part_b"

os.makedirs(output_dir_part_a, exist_ok=True)
os.makedirs(output_dir_part_b, exist_ok=True)


def read_gray_image(image_path):
    """Load an image from the specified path and convert it to a numpy array."""
    """Return a 2D numpy array of grayscale pixel values."""
    image = Image.open(image_path).convert("L")
    image_pixels = np.asarray(image, dtype=np.float32)
    return image_pixels


def save_image(image_array: np.ndarray, output_path: str):
    """Save a 2D numpy array as a grayscale image to the specified path."""
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="L")
    image.save(output_path)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape

    assert k_h % 2 == 1 and k_w % 2 == 1, "Kernel dimensions should be odd."

    padding_y = k_h // 2
    padding_x = k_w // 2

    padded_img = np.pad(
        img, ((padding_y, padding_y), (padding_x, padding_x)), mode="reflect"
    )

    output = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i : i + k_h, j : j + k_w]
            conv_value = np.sum(region * kernel)
            output[i, j] = conv_value

    return output


# Gaussian kernel (normalized)
G = (
    np.array(
        [
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1],
        ],
        dtype=np.float32,
    )
    / 273.0
)

# Sobel kernels
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


#########################
#        PART 1         #
#########################


def part1():
    building_image_pixels = read_gray_image(building_path)
    building_smoothed = convolve(building_image_pixels, G)
    save_image(building_smoothed, f"{output_dir_part_a}/2_building_smoothed.jpg")
    Gx = convolve(building_smoothed, Kx)
    save_image(Gx, f"{output_dir_part_a}/3_x_gradient.jpg")
    Gy = convolve(building_smoothed, Ky)
    save_image(Gy, f"{output_dir_part_a}/4_y_gradient.jpg")
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    gradient_magnitude_normalized = (
        gradient_magnitude / np.max(gradient_magnitude)
    ) * 255.0
    save_image(
        gradient_magnitude_normalized, f"{output_dir_part_a}/5_edge_magnitude.jpg"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(building_image_pixels, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(building_smoothed, cmap="gray")
    axes[0, 1].set_title("After Gaussian Smoothing")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(Gx, cmap="gray")
    axes[0, 2].set_title("Gradient Gx (Horizontal Edges)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(Gy, cmap="gray")
    axes[1, 0].set_title("Gradient Gy (Vertical Edges)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gradient_magnitude, cmap="gray")
    axes[1, 1].set_title("Gradient Magnitude √(Gx² + Gy²)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(gradient_magnitude_normalized, cmap="gray")
    axes[1, 2].set_title("Edge Detection Result (Normalized)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir_part_a}/6_complete_pipeline.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


#########################
#        PART 2         #
#########################
import cv2


def low_pass_filter(image, kernel_size=21, sigma=10):
    """
    Apply low-pass filter (Gaussian blur) to retain low frequencies.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian

    Returns:
        Low-pass filtered image
    """
    # Use OpenCV's Gaussian blur for efficiency
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def high_pass_filter(image, kernel_size=21, sigma=10):
    # Get low-pass version
    low_pass = low_pass_filter(image, kernel_size, sigma)

    # High-pass = Original - Low-pass
    high_pass = image.astype(np.float64) - low_pass.astype(np.float64)

    return high_pass


def create_hybrid_image(image_low, image_high, sigma_low=10, sigma_high=5):
    """
    Create a hybrid image by combining low frequencies from one image
    and high frequencies from another.

    Args:
        image_low: Image to extract low frequencies from
        image_high: Image to extract high frequencies from
        sigma_low: Sigma for low-pass filter
        sigma_high: Sigma for high-pass filter

    Returns:
        Hybrid image
    """
    # Apply low-pass to first image
    low_freq = low_pass_filter(image_low, kernel_size=31, sigma=sigma_low)

    # Apply high-pass to second image
    high_freq = high_pass_filter(image_high, kernel_size=31, sigma=sigma_high)

    # Combine them
    hybrid = low_freq + high_freq

    # Clip to valid range
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

    return hybrid, low_freq, high_freq


def ensure_same_size(image1, image2):
    print(f"Image A shape: {image1.shape}")
    print(f"Image C shape: {image2.shape}")
    if image1.shape == image2.shape:
        print("Images have the same dimensions.")
        return image1, image2

    print("Resizing images...")
    target_shape = (
        min(image1.shape[0], image2.shape[0]),
        min(image1.shape[1], image2.shape[1]),
    )
    image1 = np.array(
        Image.fromarray(image1).resize((target_shape[1], target_shape[0]))
    )
    image2 = np.array(
        Image.fromarray(image2).resize((target_shape[1], target_shape[0]))
    )
    print("After resizing:")
    print(f"Image A shape: {image1.shape}")
    print(f"Image C shape: {image2.shape}")
    return image1, image2


def part2():
    elephant_img = np.array(Image.open(elephant_path))
    cheetah_img = np.array(Image.open(cheetah_path))
    # Ensure images are the same size
    elephant_img, cheetah_img = ensure_same_size(elephant_img, cheetah_img)
    # Experiment with different sigma values
    sigma_experiments = [
        (15, 3),  # More blur in low, sharp high
        (10, 5),  # Balanced
        (5, 8),  # Less blur in low, softer high
    ]
    results = []

    for idx, (sigma_low, sigma_high) in enumerate(sigma_experiments):
        print(f"\nExperiment {idx + 1}: sigma_low={sigma_low}, sigma_high={sigma_high}")

        # Create hybrid image
        hybrid, low_freq, high_freq = create_hybrid_image(
            elephant_img, cheetah_img, sigma_low, sigma_high
        )

        results.append(
            {
                "sigma_low": sigma_low,
                "sigma_high": sigma_high,
                "hybrid": hybrid,
                "low_freq": low_freq,
                "high_freq": high_freq,
            }
        )

        # Save individual results
        Image.fromarray(low_freq.astype(np.uint8)).save(
            f"{output_dir_part_b}/{idx + 1}_low_pass_sigma_{sigma_low}.png"
        )

        # Normalize high-pass for visualization
        high_vis = high_freq - high_freq.min()
        high_vis = (high_vis / high_vis.max() * 255).astype(np.uint8)
        Image.fromarray(high_vis).save(
            f"{output_dir_part_b}/{idx + 1}_high_pass_sigma_{sigma_high}.png"
        )

        Image.fromarray(hybrid).save(
            f"{output_dir_part_b}/{idx + 1}_hybrid_{sigma_low}_{sigma_high}.png"
        )
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for idx, result in enumerate(results):
        axes[idx, 0].imshow(result["low_freq"].astype(np.uint8))
        axes[idx, 0].set_title(f"Low-pass (σ={result['sigma_low']})")
        axes[idx, 0].axis("off")

        high_vis = result["high_freq"] - result["high_freq"].min()
        high_vis = (high_vis / high_vis.max() * 255).astype(np.uint8)
        axes[idx, 1].imshow(high_vis)
        axes[idx, 1].set_title(f"High-pass (σ={result['sigma_high']})")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(result["hybrid"])
        axes[idx, 2].set_title("Hybrid Image")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir_part_b}/comparison.png", dpi=150)
    plt.close()

    # Simulate viewing at different distances
    best_hybrid = results[1]["hybrid"]  # Use balanced version

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    sizes = [1.0, 0.5, 0.25, 0.125]
    titles = ["Close (100%)", "Medium (50%)", "Far (25%)", "Very Far (12.5%)"]

    for ax, size, title in zip(axes.flat, sizes, titles):
        resized = cv2.resize(best_hybrid, None, fx=size, fy=size)
        ax.imshow(resized)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir_part_b}/distance_simulation.png", dpi=150)


def main():
    part1()
    part2()


if __name__ == "__main__":
    main()
