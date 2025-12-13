"""
Extract hand landmarks from video files using MediaPipe.

Usage:
    python src/extract_landmarks.py --input data/wlasl/videos --output data/landmarks
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_landmarks_from_video(video_path: str, hands) -> np.ndarray:
    """
    Extract 21 hand landmarks from each frame of a video.

    Args:
        video_path: Path to video file
        hands: MediaPipe Hands instance

    Returns:
        landmarks: Array of shape (T, 21, 3) where T is number of frames
                   Each landmark has (x, y, z) normalized coordinates
    """
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Take first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.append([lm.x, lm.y, lm.z])
            landmarks_sequence.append(frame_landmarks)
        else:
            # No hand detected - use zeros or skip
            landmarks_sequence.append(np.zeros((21, 3)).tolist())

    cap.release()
    return np.array(landmarks_sequence, dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks for scale and position invariance.

    Centers landmarks on wrist (landmark 0) and scales by palm size.

    Args:
        landmarks: Array of shape (T, 21, 3)

    Returns:
        Normalized landmarks of same shape
    """
    if landmarks.size == 0:
        return landmarks

    # Center on wrist (landmark 0)
    wrist = landmarks[:, 0:1, :]  # Shape: (T, 1, 3)
    centered = landmarks - wrist

    # Scale by distance from wrist to middle finger MCP (landmark 9)
    palm_size = np.linalg.norm(centered[:, 9, :], axis=1, keepdims=True)  # Shape: (T, 1)
    palm_size = np.maximum(palm_size, 1e-6)  # Avoid division by zero

    # Reshape for broadcasting
    palm_size = palm_size[:, :, np.newaxis]  # Shape: (T, 1, 1)
    normalized = centered / palm_size

    return normalized


def process_dataset(input_dir: str, output_dir: str):
    """
    Process all videos in a directory and save landmarks.

    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save .npy landmark files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))

    for video_file in tqdm(video_files, desc="Extracting landmarks"):
        landmarks = extract_landmarks_from_video(str(video_file), hands)
        landmarks = normalize_landmarks(landmarks)

        # Save as .npy file
        output_file = output_path / f"{video_file.stem}.npy"
        np.save(output_file, landmarks)

    hands.close()
    print(f"Processed {len(video_files)} videos. Landmarks saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks from videos")
    parser.add_argument("--input", required=True, help="Input video directory")
    parser.add_argument("--output", required=True, help="Output landmarks directory")
    args = parser.parse_args()

    process_dataset(args.input, args.output)
