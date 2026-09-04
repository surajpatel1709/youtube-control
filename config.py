"""
Configuration Module for YouTube Hand Gesture Controller.
Contains camera settings, detection thresholds, keyboard bindings, and UI design colors.
"""

import os

# ==========================================
# Camera & Video Stream Settings
# ==========================================
CAMERA_INDEX = 0             # Default webcam index (0 is usually the built-in or primary camera)
FRAME_WIDTH = 640            # Fast native webcam width (640x480 provides smooth 30 FPS without lag)
FRAME_HEIGHT = 480           # Fast native webcam height
MIRROR_FEED = True           # Mirror the video feed horizontally for intuitive user interaction

# ==========================================
# MediaPipe Model Settings
# ==========================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MIN_HAND_DETECTION_CONFIDENCE = 0.55
MIN_HAND_TRACKING_CONFIDENCE = 0.55
NUM_HANDS = 1                # Primary hand used for control (prevents conflicting gestures)

# ==========================================
# Gesture Detection & Motion Thresholds
# ==========================================
# Number of frames required in rolling window to confirm a gesture
GESTURE_CONFIRM_FRAMES = 3

# Buffer size for tracking hand center movement (used for swipe and velocity estimation)
MOTION_HISTORY_LENGTH = 10

# Minimum normalized displacement across history frames to register a swipe
SWIPE_THRESHOLD_X = 0.12     # Natural horizontal swipe distance (~75-90px)
SWIPE_THRESHOLD_Y = 0.11     # Natural vertical swipe distance (~50-60px)

# Action Cooldowns (in seconds) to prevent accidental rapid re-triggering
COOLDOWNS = {
    "PLAY_PAUSE": 1.5,
    "MUTE_UNMUTE": 1.5,
    "FULLSCREEN": 1.5,
    "NEXT_VIDEO": 1.5,
    "PREV_VIDEO": 1.5,
    "SEEK_FORWARD": 0.8,
    "SEEK_BACKWARD": 0.8,
    "VOLUME_UP": 0.22,
    "VOLUME_DOWN": 0.22,
    "TOGGLE_ACTIVE": 2.0,
}

# ==========================================
# YouTube Keyboard Mappings (PyAutoGUI)
# YouTube Web Player Standard Hotkeys
# ==========================================
HOTKEYS = {
    "PLAY_PAUSE": "k",                 # 'k' or 'space' toggles play/pause reliably
    "NEXT_VIDEO": ("shift", "n"),      # Shift + N jumps to next video
    "PREV_VIDEO": ("shift", "p"),      # Shift + P jumps to previous video
    "MUTE_UNMUTE": "m",                # 'm' toggles mute/unmute
    "FULLSCREEN": "f",                 # 'f' toggles browser fullscreen
    "SEEK_FORWARD": "l",               # 'l' or 'right' jumps forward 10 seconds
    "SEEK_BACKWARD": "j",              # 'j' or 'left' jumps backward 10 seconds
    "VOLUME_UP": "up",                 # Up Arrow increases volume by 5%
    "VOLUME_DOWN": "down",             # Down Arrow decreases volume by 5%
}

# ==========================================
# UI Design & Styling Colors (BGR format for OpenCV)
# Sleek Dark Cyber Aesthetic
# ==========================================
COLOR_BG_DARK = (20, 24, 30)           # Dark charcoal background
COLOR_PRIMARY = (255, 180, 0)          # Neon Cyan / Sky Blue (BGR)
COLOR_SUCCESS = (80, 220, 80)          # Vibrant Emerald Green
COLOR_WARNING = (0, 165, 255)          # Orange / Amber
COLOR_DANGER = (50, 50, 240)           # Vivid Crimson Red
COLOR_TEXT_WHITE = (245, 245, 245)     # Crisp Off-White
COLOR_TEXT_MUTED = (160, 160, 160)     # Subtle Gray
COLOR_ACCENT = (255, 90, 150)          # Purple-Pink accent
COLOR_LANDMARK_POINT = (0, 240, 255)   # Glowing yellow for joints
COLOR_LANDMARK_LINE = (220, 140, 0)    # Blue connecting bones
