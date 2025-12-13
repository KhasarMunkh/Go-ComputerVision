"""
Real-time webcam demo for sign language recognition with full 180 landmarks.

Uses: Hands (42) + Pose (6) + Face (132) = 180 landmarks

Usage:
    python src/demo_180.py --model checkpoints/sign_language_model.pt
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

from models.lstm import SignLSTM, SignLSTMWithAttention
from models.transformer import SignTransformer, SignTransformerSimple


class SignLanguageDemo180:
    """Real-time sign language recognition using 180 landmarks (hands + pose + face)."""

    # MediaPipe Face Mesh indices for the 132 landmarks used in MuteMotion dataset
    # These cover key facial features: eyes, eyebrows, nose, mouth, face contour
    FACE_LANDMARK_INDICES = [
        # Face contour (17 points)
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        # Right eyebrow (5 points)
        46, 53, 52, 65, 55,
        # Left eyebrow (5 points)
        285, 295, 282, 283, 276,
        # Nose bridge (4 points)
        6, 197, 195, 5,
        # Lower nose (5 points)
        4, 1, 19, 94, 2,
        # Right eye (6 points)
        33, 7, 163, 144, 145, 153,
        # Left eye (6 points)
        362, 382, 381, 380, 374, 373,
        # Outer lips (12 points)
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
        # Inner lips (8 points)
        78, 95, 88, 178, 87, 14, 317, 402,
        # Additional points for expression (fill to 132)
        # Cheeks
        116, 123, 147, 187, 205, 36, 345, 352, 376, 411, 425, 266,
        # Forehead
        10, 67, 109, 151, 21, 54, 103, 104, 105, 66, 107,
        296, 336, 337, 338, 284, 298, 301, 368, 264, 447,
        # More face points
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
        454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        # Fill remaining to reach 132
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
        98, 97, 326, 327, 168, 8, 9, 168, 6, 122, 351, 168,
    ]

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
        self.landmark_indices = self.config.get('landmark_indices', list(range(180)))
        self.input_dim = self.config.get('input_dim', 540)
        self.num_landmarks = len(self.landmark_indices)
        print(f"Model expects {self.num_landmarks} landmarks (input_dim={self.input_dim})")

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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

        # Face mesh detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
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

        # Visualization settings
        self.show_face = True
        self.show_hands = True
        self.show_pose = True

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get config
        config = checkpoint.get('config', {})

        # Get model type
        model_type = config.get('model_type', 'lstm_attention')
        print(f"Model type: {model_type}")

        # Get dimensions
        input_dim = config.get('input_dim', 540)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        num_classes = checkpoint.get('num_classes', len(checkpoint.get('idx_to_gloss', {})))

        print(f"Num classes: {num_classes}")
        print(f"Input dim: {input_dim}")

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
        if idx_to_gloss and isinstance(list(idx_to_gloss.keys())[0], str):
            idx_to_gloss = {int(k): v for k, v in idx_to_gloss.items()}

        print(f"Loaded {len(idx_to_gloss)} sign classes")
        print(f"Example signs: {list(idx_to_gloss.values())[:5]}")

        return model, config, idx_to_gloss

    def _extract_landmarks(self, frame) -> np.ndarray:
        """
        Extract all 180 landmarks matching the training data format.

        Returns array of shape (180, 3) or None if no detection.
        Format: [Right Hand (21), Left Hand (21), Pose (6), Face (132)] = 180 landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize with zeros: 180 landmarks x 3 coords
        all_landmarks = np.zeros((180, 3), dtype=np.float32)

        has_detection = False

        # Get hand landmarks (indices 0-41)
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            has_detection = True
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[i].classification[0].label
                hand_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                if handedness == 'Right':
                    all_landmarks[0:21] = hand_lms  # Right hand: indices 0-20
                else:
                    all_landmarks[21:42] = hand_lms  # Left hand: indices 21-41

        # Get pose landmarks - upper body (indices 42-47)
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            has_detection = True
            pose_lms = pose_results.pose_landmarks.landmark
            # Upper body: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
            pose_indices = [11, 12, 13, 14, 15, 16]
            for j, idx in enumerate(pose_indices):
                lm = pose_lms[idx]
                all_landmarks[42 + j] = [lm.x, lm.y, lm.z]

        # Get face landmarks (indices 48-179)
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            has_detection = True
            face_landmarks = face_results.multi_face_landmarks[0]

            # Extract 132 face landmarks
            # MediaPipe Face Mesh has 478 landmarks, we select 132 key ones
            for j in range(132):
                if j < len(self.FACE_LANDMARK_INDICES):
                    mp_idx = self.FACE_LANDMARK_INDICES[j]
                    if mp_idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[mp_idx]
                        all_landmarks[48 + j] = [lm.x, lm.y, lm.z]

        if not has_detection:
            return None

        # Select only the landmarks the model expects
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

    def _draw_landmarks(self, frame, rgb_frame):
        """Draw all detected landmarks on frame."""
        h, w = frame.shape[:2]

        # Draw hands
        if self.show_hands:
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

        # Draw pose (upper body only)
        if self.show_pose:
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                # Draw only upper body connections
                upper_body_connections = [
                    (11, 12),  # shoulders
                    (11, 13), (13, 15),  # left arm
                    (12, 14), (14, 16),  # right arm
                ]
                for start_idx, end_idx in upper_body_connections:
                    start = pose_results.pose_landmarks.landmark[start_idx]
                    end = pose_results.pose_landmarks.landmark[end_idx]
                    start_pt = (int(start.x * w), int(start.y * h))
                    end_pt = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_pt, end_pt, (255, 255, 0), 2)

                # Draw pose landmarks
                for idx in [11, 12, 13, 14, 15, 16]:
                    lm = pose_results.pose_landmarks.landmark[idx]
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 5, (255, 255, 0), -1)

        # Draw face mesh
        if self.show_face:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                # Draw face contour and key features with light styling
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(200, 200, 200), thickness=1
                    )
                )
                # Highlight lips (important for many signs)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 150, 255), thickness=2
                    )
                )

    def run(self):
        """Run the webcam demo."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*50)
        print("SIGN LANGUAGE RECOGNITION DEMO (180 Landmarks)")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear buffer")
        print("  'space' - Pause/Resume")
        print("  'f' - Toggle face landmarks")
        print("  'h' - Toggle hand landmarks")
        print("  'p' - Toggle pose landmarks")
        print("="*50 + "\n")

        paused = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not paused:
                # Extract landmarks
                landmarks = self._extract_landmarks(frame)

                if landmarks is not None:
                    self.landmark_buffer.append(landmarks)
                    self.current_prediction, self.confidence, self.top5_predictions = self._predict()

            # Draw landmarks
            self._draw_landmarks(frame, rgb_frame)

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

            # Landmark toggle indicators
            toggles = []
            if self.show_hands:
                toggles.append("H")
            if self.show_pose:
                toggles.append("P")
            if self.show_face:
                toggles.append("F")
            cv2.putText(frame, f"[{'/'.join(toggles)}]", (500, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

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

            # Model info
            cv2.putText(frame, f"180 landmarks", (520, 475),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.imshow('Sign Language Recognition (180 Landmarks)', frame)

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
            elif key == ord('f'):
                self.show_face = not self.show_face
                print(f"Face landmarks: {'ON' if self.show_face else 'OFF'}")
            elif key == ord('h'):
                self.show_hands = not self.show_hands
                print(f"Hand landmarks: {'ON' if self.show_hands else 'OFF'}")
            elif key == ord('p'):
                self.show_pose = not self.show_pose
                print(f"Pose landmarks: {'ON' if self.show_pose else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()
        self.face_mesh.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language recognition demo (180 landmarks)")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    demo = SignLanguageDemo180(args.model, args.device)
    demo.run()


if __name__ == "__main__":
    main()
