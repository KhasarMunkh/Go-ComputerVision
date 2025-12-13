"""
Real-time webcam demo for sign language recognition.

Usage:
    python src/demo.py --model checkpoints/sign_language_model.pt
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

from models.lstm import SignLSTM, SignLSTMWithAttention
from models.transformer import SignTransformer, SignTransformerSimple


class SignLanguageDemo:
    """Real-time sign language recognition from webcam."""

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize demo.

        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Load model
        self.model, self.config, self.idx_to_gloss = self._load_model(model_path)
        self.model.eval()

        # Get landmark configuration from model
        self.landmark_indices = self.config.get('landmark_indices', list(range(48)))
        self.input_dim = self.config.get('input_dim', 144)
        self.num_landmarks = len(self.landmark_indices)
        print(f"Model expects {self.num_landmarks} landmarks (input_dim={self.input_dim})")

        # Initialize MediaPipe for hands and pose
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Two hands detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pose detection (for upper body landmarks)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Buffer for landmark sequence
        self.max_seq_length = self.config.get('max_seq_length', 60)
        self.landmark_buffer = deque(maxlen=self.max_seq_length)

        # State
        self.current_prediction = "Waiting..."
        self.confidence = 0.0
        self.top5_predictions = []

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get config - handle both old and new checkpoint formats
        config = checkpoint.get('config', {})

        # Get model type
        model_type = config.get('model_type', 'lstm_attention')
        print(f"Model type: {model_type}")

        # Get dimensions
        input_dim = config.get('input_dim', 144)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        num_classes = checkpoint.get('num_classes', 100)

        print(f"Num classes: {num_classes}")

        # Create model
        if model_type == 'lstm':
            model = SignLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
        elif model_type == 'lstm_attention':
            model = SignLSTMWithAttention(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
        elif model_type == 'transformer':
            model = SignTransformer(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            model = SignTransformerSimple(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # Load class mapping
        idx_to_gloss = checkpoint.get('idx_to_gloss', {i: f"sign_{i}" for i in range(num_classes)})
        # Convert keys to int if they're strings
        if idx_to_gloss and isinstance(list(idx_to_gloss.keys())[0], str):
            idx_to_gloss = {int(k): v for k, v in idx_to_gloss.items()}

        print(f"Loaded {len(idx_to_gloss)} sign classes")
        print(f"Example signs: {list(idx_to_gloss.values())[:5]}")

        return model, config, idx_to_gloss

    def _extract_landmarks(self, frame) -> np.ndarray:
        """
        Extract landmarks matching the training data format.

        Returns array of shape (num_landmarks, 3) or None if no detection.
        Format: [Right Hand (21), Left Hand (21), Pose (6)] = 48 landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize with zeros
        # Full format: Right Hand (21) + Left Hand (21) + Pose (6) + Face (132) = 180
        # We use: Right Hand (21) + Left Hand (21) + Pose (6) = 48
        all_landmarks = np.zeros((180, 3), dtype=np.float32)

        # Get hand landmarks
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Determine if left or right hand
                handedness = hand_results.multi_handedness[i].classification[0].label

                hand_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                if handedness == 'Right':
                    all_landmarks[0:21] = hand_lms  # Right hand: indices 0-20
                else:
                    all_landmarks[21:42] = hand_lms  # Left hand: indices 21-41

        # Get pose landmarks (upper body)
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            pose_lms = pose_results.pose_landmarks.landmark
            # Upper body landmarks: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
            pose_indices = [11, 12, 13, 14, 15, 16]
            for j, idx in enumerate(pose_indices):
                lm = pose_lms[idx]
                all_landmarks[42 + j] = [lm.x, lm.y, lm.z]

        # Check if we have any hand detection
        has_hands = np.any(all_landmarks[0:42] != 0)

        if not has_hands:
            return None

        # Select only the landmarks we need (first 48 for hands + pose)
        selected_landmarks = all_landmarks[self.landmark_indices]

        return selected_landmarks

    def _predict(self) -> tuple:
        """Make prediction from current landmark buffer."""
        if len(self.landmark_buffer) < 10:  # Need minimum frames
            return "Collecting frames...", 0.0, []

        # Prepare input
        landmarks = np.array(list(self.landmark_buffer))  # (T, num_landmarks, 3)
        landmarks = landmarks.reshape(landmarks.shape[0], -1)  # (T, num_landmarks*3)

        # Pad or sample to max_seq_length
        T = landmarks.shape[0]
        if T > self.max_seq_length:
            # Uniform sampling
            indices = np.linspace(0, T - 1, self.max_seq_length, dtype=int)
            landmarks = landmarks[indices]
        elif T < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - T, landmarks.shape[1]))
            landmarks = np.vstack([landmarks, padding])

        # To tensor
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(1)

            # Get top 5 predictions
            top5_probs, top5_indices = probs.topk(5, dim=1)
            top5 = [(self.idx_to_gloss.get(idx.item(), "?"), prob.item())
                    for idx, prob in zip(top5_indices[0], top5_probs[0])]

        pred_label = self.idx_to_gloss.get(pred_idx.item(), f"sign_{pred_idx.item()}")
        return pred_label, confidence.item(), top5

    def run(self):
        """Run the webcam demo."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*50)
        print("SIGN LANGUAGE RECOGNITION DEMO")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear buffer")
        print("  'space' - Pause/Resume")
        print("="*50 + "\n")

        paused = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            if not paused:
                # Extract landmarks
                landmarks = self._extract_landmarks(frame)

                if landmarks is not None:
                    self.landmark_buffer.append(landmarks)
                    self.current_prediction, self.confidence, self.top5_predictions = self._predict()

            # Draw hand landmarks for visualization
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

            # Draw UI - Top bar
            cv2.rectangle(frame, (0, 0), (640, 100), (40, 40, 40), -1)

            # Main prediction
            color = (0, 255, 0) if self.confidence > 0.3 else (0, 255, 255)
            cv2.putText(frame, f"{self.current_prediction}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"Confidence: {self.confidence:.1%}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Buffer indicator
            buffer_pct = len(self.landmark_buffer) / self.max_seq_length
            cv2.rectangle(frame, (500, 20), (630, 40), (100, 100, 100), -1)
            cv2.rectangle(frame, (500, 20), (int(500 + 130 * buffer_pct), 40), (0, 255, 0), -1)
            cv2.putText(frame, f"{len(self.landmark_buffer)}/{self.max_seq_length}",
                        (545, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Pause indicator
            if paused:
                cv2.putText(frame, "PAUSED", (500, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Top 5 predictions sidebar
            if self.top5_predictions:
                cv2.rectangle(frame, (0, 380), (200, 480), (40, 40, 40), -1)
                cv2.putText(frame, "Top 5:", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                for i, (sign, prob) in enumerate(self.top5_predictions):
                    y = 420 + i * 15
                    cv2.putText(frame, f"{sign}: {prob:.1%}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow('Sign Language Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.landmark_buffer.clear()
                self.current_prediction = "Buffer cleared"
                self.confidence = 0.0
                self.top5_predictions = []
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language recognition demo")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    demo = SignLanguageDemo(args.model, args.device)
    demo.run()


if __name__ == "__main__":
    main()
