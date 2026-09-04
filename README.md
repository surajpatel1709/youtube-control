# 🖐️ YouTube Hand Gesture Controller

A high-performance, real-time computer vision application that allows you to control YouTube playback in any web browser without touching your mouse or keyboard, using intuitive hand gestures captured by your PC/laptop webcam.

Built with **Python**, **OpenCV**, **MediaPipe**, **PyAutoGUI**, and **NumPy**.

---

## 🌟 Key Features

* **Touchless Video Control**: Play, pause, adjust volume, seek forward/backward, skip videos, mute, and toggle fullscreen using natural hand gestures.
* **Dual-Engine MediaPipe Compatibility**: Fully compatible with the modern **MediaPipe Tasks Vision API** (`HandLandmarker`) as well as legacy `mp.solutions.hands`.
* **Smart Gesture Disambiguation**: Differentiates between seeking inside a video (10s arrow gestures) and switching videos (swipe wave gestures).
* **Anti-Flicker & Debounce Engine**: Employs multi-frame state confirmation ($N$ consecutive frames) to eliminate false triggers, paired with per-action cooldowns.
* **Cyber Heads-Up Display (HUD)**: Sleek, translucent on-screen display showing real-time FPS, detected gesture, active action flash banner, cooldown status, and a quick-reference cheat sheet.
* **Sleep / Safe Mode**: Toggle gesture control on/off via a dedicated "Call-Me" hand sign or spacebar, allowing you to move freely without accidentally triggering commands.
* **Zero-Touch Setup**: Automatically downloads the required MediaPipe model (`hand_landmarker.task`) on first run if not already present.

---

## 📁 Project Architecture

```text
youtube_gesture_controller/
│
├── config.py                # Central configuration (thresholds, keys, timeouts, colors, cooldowns)
├── gesture_detector.py      # MediaPipe hand tracking, landmark analysis, pose & swipe recognition
├── youtube_controller.py    # PyAutoGUI action executor with cooldowns, debouncing & safety
├── ui_overlay.py            # Translucent Cyber HUD rendering (status cards, FPS, badges, cheat sheet)
├── main.py                  # Main loop, video capture, orchestration, and keyboard handling
├── test_controller.py       # Standalone test/diagnostics suite (runs without camera)
├── hand_landmarker.task     # MediaPipe Hand Landmarker neural model asset
├── requirements.txt         # Project dependencies
└── README.md                # Complete documentation and user manual
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.9, 3.10, 3.11, 3.12, or 3.13
* A working webcam (built-in or USB external)
* Google Chrome, Mozilla Firefox, Microsoft Edge, Brave, or any modern web browser

### 2. Install Dependencies
Open your command prompt or terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## 🎮 How to Run

### Step 1: Open YouTube in your Web Browser
1. Open your browser and navigate to any video on [YouTube](https://www.youtube.com).
2. Click anywhere on the YouTube player page so that the browser window has focus.

### Step 2: Start the Gesture Controller
Open a terminal in this directory and execute:

```bash
python main.py
```

A webcam window titled **"YouTube Hand Gesture Controller"** will appear with the live video feed and Cyber HUD.

### Keyboard Shortcuts in the Webcam Window
* `Q` or `ESC` : Safely quit the application and release the webcam.
* `Spacebar`   : Toggle gesture control between **ACTIVE** and **PAUSED** (sleep mode).

---

## ✋ Supported Hand Gestures

| Gesture | Hand Pose | YouTube Action | Hotkey Triggered | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Open Palm (Hold)** | All 5 fingers extended steadily | **Play / Pause** | `K` / `Space` | Toggles video playback. |
| **Palm Swipe Left** | Open palm moved swiftly left | **Next Video** | `Shift + N` | Skips to the next video in queue/playlist. |
| **Palm Swipe Right** | Open palm moved swiftly right | **Previous Video** | `Shift + P` | Returns to the previous video. |
| **Point Swipe Right** | Single index finger pointing right | **Seek Forward (10s)** | `L` / `Right Arrow` | Skips forward by 10 seconds. |
| **Point Swipe Left** | Single index finger pointing left | **Seek Backward (10s)** | `J` / `Left Arrow` | Skips backward by 10 seconds. |
| **Point Hand Up** | Index finger pointing upwards | **Volume Up** | `Up Arrow` | Increases volume in 5% increments. |
| **Point Hand Down** | Index finger pointing downwards | **Volume Down** | `Down Arrow` | Decreases volume in 5% increments. |
| **Closed Fist** | All fingers curled into palm | **Mute / Unmute** | `M` | Toggles audio mute state. |
| **Victory / Peace (V)** | Index & Middle fingers extended | **Fullscreen** | `F` | Toggles theater/fullscreen view. |
| **"Call Me" / Shaka** | Thumb & Pinky extended, others curled | **Toggle Control** | *Internal State* | Toggles between ACTIVE and PAUSED. |

---

## 🧠 How MediaPipe Hand Landmarks Work

MediaPipe tracks **21 distinct 3D landmarks** ($x, y, z$) for each hand in real time:

```
        8   12  16  20       Tips:
        |   |   |   |        4  = Thumb Tip
    4   7   11  15  19       8  = Index Tip
    |   |   |   |   |        12 = Middle Tip
    3   6   10  14  18       16 = Ring Tip
    |   |   |   |   |        20 = Pinky Tip
    2   5---9---13--17       MCP Joints:
     \ /             |       5, 9, 13, 17
      1              |       Base / Wrist:
       \            /        0  = Wrist
         -----0-----
```

### 1. Rotation-Invariant Geometric Analysis
Unlike naive implementations that simply compare $y$-coordinates (which break whenever the hand is tilted or horizontal), our detector uses **Euclidean distance ratios**:
* A finger is detected as **EXTENDED** if:
  $$\text{dist}(\text{Tip}, \text{Wrist}) > 1.10 \times \text{dist}(\text{PIP}, \text{Wrist}) \quad\text{and}\quad \text{dist}(\text{Tip}, \text{MCP}) > 1.10 \times \text{dist}(\text{PIP}, \text{MCP})$$
* A finger is detected as **CURLED** when its tip folds backward toward the MCP base.
* This allows accurate detection whether your hand is upright, angled toward the camera, or tilted sideways.

### 2. Swipe & Velocity Vector Tracking
* The detector maintains a rolling ring buffer of hand centroids $(\bar{x}, \bar{y})$ over a 12-frame history window.
* When a motion is detected, horizontal displacement $\Delta x$ and vertical displacement $\Delta y$ are calculated:
  * Horizontal swipe: $|\Delta x| > \text{Threshold}_X$ and $|\Delta x| > 1.35 \times |\Delta y|$.
  * Vertical movement: $|\Delta y| > \text{Threshold}_Y$ and $|\Delta y| > 1.35 \times |\Delta x|$.

### 3. Multi-Frame Confirmation & Cooldowns
* **Debouncing**: To prevent accidental single-frame false alarms, static gestures must be detected consistently for `GESTURE_CONFIRM_FRAMES` (default: 4 frames) before an action is dispatched.
* **Per-Action Cooldowns**: Once triggered, each action enters an independent cooldown timer (e.g., 1.5s for Play/Pause, 0.22s for smooth volume increments).

---

## ⚙️ Configuration (`config.py`)

All parameters are easily customizable in [config.py](file:///c:/Users/ACER/Desktop/youtube%20control/config.py):

* **Camera Resolution**: `FRAME_WIDTH = 1280`, `FRAME_HEIGHT = 720`.
* **Mirroring**: `MIRROR_FEED = True` (flips feed horizontally so moving your hand right corresponds to screen right).
* **Sensitivity**:
  * `SWIPE_THRESHOLD_X`: Horizontal swipe sensitivity (default `0.16`).
  * `SWIPE_THRESHOLD_Y`: Vertical movement sensitivity (default `0.14`).
  * `GESTURE_CONFIRM_FRAMES`: Number of stable frames required to confirm a pose (default `4`).
* **Hotkeys**: Modify hotkeys in `HOTKEYS` dictionary to map to different applications (e.g., VLC, Netflix, Spotify).

---

## 🧪 Diagnostics & Offline Testing

To test and verify the entire system without opening a camera, run the included test suite:

```bash
python test_controller.py
```

This verifies:
1. MediaPipe model asset integrity.
2. Detector initialization with dummy frames.
3. Geometric landmark logic against mock poses (Open Palm, Closed Fist, Peace Sign).
4. PyAutoGUI multi-frame debouncing and cooldown mechanisms.
5. Cyber HUD rendering pipeline.

---

## 🛠️ Troubleshooting Guide

### 1. "Could not open any webcam"
* Ensure your webcam is properly connected and not being used by another application (e.g., Zoom, Teams, Skype, or Windows Camera app).
* If you have multiple cameras, change `CAMERA_INDEX` in `config.py` from `0` to `1` or `2`.
* On Windows, check **Settings > Privacy & Security > Camera** to ensure desktop apps have permission to access the webcam.

### 2. Actions Not Affecting YouTube
* PyAutoGUI sends keystrokes to the **active frontmost window**. Ensure you click on the YouTube web browser tab before performing gestures so that it holds keyboard focus.
* Make sure YouTube player focus is active (clicking inside the video once gives the player direct key control).

### 3. MediaPipe / Solutions AttributeError
* If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, this occurs on newer MediaPipe versions (0.10.20+).
* Our application is built specifically to handle this using the **MediaPipe Tasks API** (`HandLandmarker`) and bundles `hand_landmarker.task`.

### 4. Gestures Triggering Too Quickly or Slowly
* If gestures trigger too quickly, increase `GESTURE_CONFIRM_FRAMES` (e.g., from `4` to `6`) in `config.py`.
* If volume changes too fast, increase `"VOLUME_UP"` cooldown in `config.py` from `0.22` to `0.40`.

---

## 🔮 Future Enhancements

* **Two-Hand Control**: Use two hands simultaneously (e.g., one hand for playback, second hand for continuous volume pinch-slider).
* **Continuous Pinch-to-Seek**: Pinch thumb and index finger together and slide horizontally like an on-screen scrubber bar.
* **System Brightness & Media Integration**: Direct Windows OS volume/brightness sliders via `pycaw` / `screen-brightness-control`.
* **Voice + Gesture Hybrid**: Combine simple voice commands ("Hey YouTube", "Search") with gesture shortcuts.
* **Multi-Platform Support**: Presets for Netflix, Spotify Desktop, VLC Media Player, and Disney+.
