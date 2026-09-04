"""
Gesture Detector Module for YouTube Hand Gesture Controller.
Handles MediaPipe Hand Landmarker initialization, landmark extraction,
geometric finger state analysis, swipe velocity tracking, and gesture classification.
"""

import math
import time
import urllib.request
import os
from collections import deque
import cv2
import numpy as np

import config


class GestureDetector:
    """
    Robust hand landmark and gesture recognition engine using Google MediaPipe.
    Compatible with MediaPipe Tasks API (modern 0.10.x+) and legacy mp.solutions fallback.
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),       # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),     # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (0, 17)                                    # Palm base
    ]

    def __init__(self, model_path=config.MODEL_PATH):
        self.model_path = model_path
        self._ensure_model_available()
        
        self.use_tasks_api = False
        self.landmarker = None
        self.legacy_hands = None
        self._init_mediapipe()

        # Motion tracking history for swipe calculations: stores tuples of (time, center_x, center_y)
        self.motion_history = deque(maxlen=config.MOTION_HISTORY_LENGTH)
        self.last_swipe_time = 0.0

    def _ensure_model_available(self):
        """Ensures the hand_landmarker.task model file exists locally; downloads if missing."""
        if not os.path.exists(self.model_path):
            print(f"[INFO] Model not found at '{self.model_path}'. Downloading MediaPipe model...")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
                urllib.request.urlretrieve(config.MODEL_URL, self.model_path)
                print(f"[INFO] Model successfully downloaded to {self.model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download model: {e}")

    def _init_mediapipe(self):
        """Initializes MediaPipe using Tasks API if available, else falls back to mp.solutions."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=config.NUM_HANDS,
                min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
                min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.IMAGE
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            print("[INFO] MediaPipe Tasks HandLandmarker initialized successfully.")
            return
        except Exception as e:
            print(f"[WARN] MediaPipe Tasks initialization failed or not supported: {e}")

        # Fallback to legacy mp.solutions.hands if available
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=config.NUM_HANDS,
                    min_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE
                )
                self.use_tasks_api = False
                print("[INFO] MediaPipe legacy solutions.hands initialized as fallback.")
                return
        except Exception as err:
            print(f"[ERROR] Failed to initialize legacy MediaPipe hands: {err}")

        raise RuntimeError("Could not initialize MediaPipe Hand Landmarker. Check installation.")

    def find_hands(self, frame_bgr):
        """
        Processes a BGR video frame and extracts 3D hand landmarks.
        Returns a list of hands, each containing a list of 21 landmark objects with .x, .y, .z.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            if result and result.hand_landmarks:
                return result.hand_landmarks
            return []
        else:
            result = self.legacy_hands.process(rgb_frame)
            if result and result.multi_hand_landmarks:
                return [hand.landmark for hand in result.multi_hand_landmarks]
            return []

    @staticmethod
    def get_pixel_landmarks(landmarks, frame_shape):
        """Converts normalized landmarks [0, 1] into integer pixel coordinates (x, y)."""
        h, w = frame_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    @staticmethod
    def _euclidean_distance(pt1, pt2):
        """Calculates 2D Euclidean distance between two landmark objects or (x, y) tuples."""
        x1 = pt1.x if hasattr(pt1, 'x') else pt1[0]
        y1 = pt1.y if hasattr(pt1, 'y') else pt1[1]
        x2 = pt2.x if hasattr(pt2, 'x') else pt2[0]
        y2 = pt2.y if hasattr(pt2, 'y') else pt2[1]
        return math.hypot(x1 - x2, y1 - y2)

    def get_finger_states(self, landmarks):
        """
        Determines whether each of the 5 fingers is EXTENDED (True) or CURLED (False).
        Uses adaptive geometric criteria combining distance from wrist and palm center.
        
        Returns:
            list[bool]: [thumb, index, middle, ring, pinky]
        """
        wrist = landmarks[0]
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcps = [2, 5, 9, 13, 17]

        extended = []

        # 1. Palm scale reference (distance across palm from Index MCP to Pinky MCP)
        palm_width = self._euclidean_distance(landmarks[5], landmarks[17])
        if palm_width < 1e-4:
            palm_width = 0.1

        # Check if hand is roughly upright (wrist y is lower on screen than middle MCP y)
        is_upright = wrist.y > landmarks[9].y

        # 2. Thumb:
        # Distance from thumb tip (4) to middle finger MCP (9) indicates if thumb is spread or tucked
        d_thumb_palm = self._euclidean_distance(landmarks[4], landmarks[9])
        d_thumb_wrist = self._euclidean_distance(landmarks[4], wrist)
        d_thumb_ip_wrist = self._euclidean_distance(landmarks[3], wrist)

        thumb_extended = (d_thumb_palm > 0.65 * palm_width) and (d_thumb_wrist > d_thumb_ip_wrist * 1.02)
        extended.append(bool(thumb_extended))

        # 3. Four fingers (Index, Middle, Ring, Pinky):
        for tip_idx, pip_idx, mcp_idx in zip(finger_tips[1:], finger_pips[1:], finger_mcps[1:]):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            d_tip_wrist = self._euclidean_distance(tip, wrist)
            d_pip_wrist = self._euclidean_distance(pip, wrist)
            d_tip_mcp = self._euclidean_distance(tip, mcp)
            d_pip_mcp = self._euclidean_distance(pip, mcp)

            # Finger is extended if tip is further from wrist and MCP than PIP
            distance_check = (d_tip_wrist > d_pip_wrist * 1.03) and (d_tip_mcp > d_pip_mcp * 1.03)

            # If upright, tip must also be higher on screen than PIP
            if is_upright:
                is_ext = distance_check or (tip.y < pip.y and d_tip_wrist > d_pip_wrist)
            else:
                is_ext = distance_check

            extended.append(bool(is_ext))

        return extended

    def get_palm_center(self, landmarks):
        """Calculates normalized (cx, cy) coordinates of palm center using wrist and MCP joints."""
        pts = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
        return cx, cy

    def update_motion_history(self, palm_center):
        """Updates the motion history ring buffer with the latest palm position and timestamp."""
        now = time.time()
        self.motion_history.append((now, palm_center[0], palm_center[1]))

    def clear_motion_history(self):
        """Clears motion history buffer when no hand is detected."""
        self.motion_history.clear()

    def detect_swipe(self):
        """
        Detects horizontal or vertical swipe movements based on recent motion history.
        Returns:
            str or None: 'SWIPE_LEFT', 'SWIPE_RIGHT', 'SWIPE_UP', 'SWIPE_DOWN', or None
        """
        if len(self.motion_history) < 5:
            return None

        now = time.time()
        if now - self.last_swipe_time < 0.5:  # Prevent rapid duplicate swipe firing
            return None

        # Compare oldest recorded point with current point in history
        oldest_time, old_x, old_y = self.motion_history[0]
        _, curr_x, curr_y = self.motion_history[-1]

        dx = curr_x - old_x
        dy = curr_y - old_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Check for dominant horizontal swipe
        if abs_dx > config.SWIPE_THRESHOLD_X and abs_dx > 1.25 * abs_dy:
            self.last_swipe_time = now
            self.motion_history.clear()
            return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"

        # Check for dominant vertical swipe
        if abs_dy > config.SWIPE_THRESHOLD_Y and abs_dy > 1.25 * abs_dx:
            self.last_swipe_time = now
            self.motion_history.clear()
            # Note: in computer vision, y decreases upward (0 at top, 1 at bottom)
            return "SWIPE_UP" if dy < 0 else "SWIPE_DOWN"

        return None

    def classify_gesture(self, landmarks):
        """
        Classifies hand landmarks into a specific recognized gesture.
        Combines static finger configurations with dynamic swipe trajectory analysis.

        Returns:
            dict with:
                - 'gesture': Name of detected gesture
                - 'action': Mapped YouTube action or None
                - 'finger_states': List of 5 bools [T, I, M, R, P]
                - 'palm_center': (cx, cy)
        """
        if not landmarks:
            self.clear_motion_history()
            return None

        # Get finger extension states: [Thumb, Index, Middle, Ring, Pinky]
        finger_states = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = finger_states
        extended_count = sum(finger_states)
        four_fingers = [index, middle, ring, pinky]
        four_extended_count = sum(four_fingers)

        palm_cx, palm_cy = self.get_palm_center(landmarks)
        self.update_motion_history((palm_cx, palm_cy))

        # Check dynamic swipe motions first
        swipe = self.detect_swipe()

        # ==========================================
        # 1. SWIPE GESTURES
        # ==========================================
        # Pointing Swipe (Seek 10s Backward/Forward)
        if index and not middle and not ring and not pinky:
            if swipe == "SWIPE_LEFT":
                return {
                    "gesture": "POINT_SWIPE_LEFT",
                    "action": "SEEK_BACKWARD",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_RIGHT":
                return {
                    "gesture": "POINT_SWIPE_RIGHT",
                    "action": "SEEK_FORWARD",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_UP":
                return {
                    "gesture": "POINT_MOVE_UP",
                    "action": "VOLUME_UP",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_DOWN":
                return {
                    "gesture": "POINT_MOVE_DOWN",
                    "action": "VOLUME_DOWN",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }

        # Open Palm Swipe (Next Video / Previous Video)
        if four_extended_count >= 3:
            if swipe == "SWIPE_LEFT":
                return {
                    "gesture": "PALM_SWIPE_LEFT",
                    "action": "NEXT_VIDEO",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_RIGHT":
                return {
                    "gesture": "PALM_SWIPE_RIGHT",
                    "action": "PREV_VIDEO",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_UP":
                return {
                    "gesture": "PALM_SWIPE_UP",
                    "action": "VOLUME_UP",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            if swipe == "SWIPE_DOWN":
                return {
                    "gesture": "PALM_SWIPE_DOWN",
                    "action": "VOLUME_DOWN",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }

        # ==========================================
        # 2. STATIC GESTURES
        # ==========================================

        # A. Disable/Enable Gesture Control ("Call Me" sign: Thumb & Pinky extended, middle 3 curled)
        if thumb and pinky and not middle and not ring:
            return {
                "gesture": "CALL_ME_TOGGLE",
                "action": "TOGGLE_ACTIVE",
                "finger_states": finger_states,
                "palm_center": (palm_cx, palm_cy)
            }

        # B. Fullscreen (Victory / Peace sign: Index and Middle extended, Ring and Pinky curled)
        if index and middle and not ring and not pinky:
            return {
                "gesture": "TWO_FINGERS_V",
                "action": "FULLSCREEN",
                "finger_states": finger_states,
                "palm_center": (palm_cx, palm_cy)
            }

        # C. Mute / Unmute (Closed Fist: All 4 main fingers curled)
        if four_extended_count == 0:
            return {
                "gesture": "CLOSED_FIST",
                "action": "MUTE_UNMUTE",
                "finger_states": finger_states,
                "palm_center": (palm_cx, palm_cy)
            }

        # D. Open Palm (At least 4 fingers extended: Play/Pause)
        if four_extended_count >= 3 and extended_count >= 4:
            return {
                "gesture": "OPEN_PALM",
                "action": "PLAY_PAUSE",
                "finger_states": finger_states,
                "palm_center": (palm_cx, palm_cy)
            }

        # E. Pointing Up / Down (Single index finger extended)
        if index and not middle and not ring and not pinky:
            index_tip_y = landmarks[8].y
            wrist_y = landmarks[0].y
            if index_tip_y < wrist_y - 0.12:
                return {
                    "gesture": "POINT_UP",
                    "action": "VOLUME_UP",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            elif index_tip_y > wrist_y + 0.05:
                return {
                    "gesture": "POINT_DOWN",
                    "action": "VOLUME_DOWN",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }
            else:
                return {
                    "gesture": "POINTING_STEADY",
                    "action": None,
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }

        # F. Thumbs Up (Thumb extended, 4 fingers curled, thumb tip above wrist)
        if thumb and four_extended_count == 0:
            if landmarks[4].y < landmarks[0].y - 0.08:
                return {
                    "gesture": "THUMB_UP",
                    "action": "PLAY_PAUSE",
                    "finger_states": finger_states,
                    "palm_center": (palm_cx, palm_cy)
                }

        # Neutral / In-between pose
        return {
            "gesture": "NEUTRAL",
            "action": None,
            "finger_states": finger_states,
            "palm_center": (palm_cx, palm_cy)
        }

    def draw_landmarks(self, frame, landmarks):
        """
        Renders sleek cyberpunk-styled hand landmarks and bone connections on the OpenCV frame.
        """
        pts = self.get_pixel_landmarks(landmarks, frame.shape)

        # Draw bone connections
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            pt1 = pts[p1_idx]
            pt2 = pts[p2_idx]
            cv2.line(frame, pt1, pt2, config.COLOR_LANDMARK_LINE, 2, cv2.LINE_AA)

        # Draw landmark joints with glowing dual circles
        for idx, (x, y) in enumerate(pts):
            # Key points (tips and wrist) are highlighted larger
            if idx in [0, 4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 7, config.COLOR_LANDMARK_POINT, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 4, (0, 210, 255), -1, cv2.LINE_AA)
