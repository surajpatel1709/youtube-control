"""
Main Application Entry Point for YouTube Hand Gesture Controller.
Initializes the webcam feed, runs real-time MediaPipe hand tracking,
evaluates gestures, dispatches YouTube commands, and renders the Cyber HUD.
"""

import sys
import time
import cv2

import config
from gesture_detector import GestureDetector
from youtube_controller import YouTubeController
import ui_overlay


def open_camera():
    """
    Attempts to open the configured webcam.
    If the default index fails, scans alternative device indices (0, 1, 2).
    """
    indices_to_try = [config.CAMERA_INDEX, 0, 1, 2]
    # Remove duplicates preserving order
    seen = set()
    indices = [x for x in indices_to_try if not (x in seen or seen.add(x))]

    for idx in indices:
        print(f"[INFO] Attempting to open camera at index {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        if cap.isOpened():
            # Test frame read
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[INFO] Successfully opened camera index {idx} ({actual_w}x{actual_h}).")
                return cap
            cap.release()

    return None


def main():
    print("=" * 65)
    print("      YOUTUBE HAND GESTURE CONTROLLER - INITIALIZING")
    print("=" * 65)
    print("[1] Make sure your browser has a YouTube video open & active.")
    print("[2] Position your hand clearly in front of the webcam.")
    print("[3] Press 'Q' or ESC at any time to exit.")
    print("[4] Press 'Space' in the webcam window to toggle pause/resume.")
    print("-" * 65)

    # 1. Initialize Camera
    cap = open_camera()
    if not cap:
        print("[ERROR] Could not open any webcam. Please ensure a camera is connected,")
        print("        accessible by Windows, and not in use by another application.")
        return 1

    # 2. Initialize MediaPipe Detector and YouTube Controller
    try:
        detector = GestureDetector()
    except Exception as e:
        print(f"[CRITICAL] Error initializing GestureDetector: {e}")
        cap.release()
        return 1

    controller = YouTubeController()

    window_name = "YouTube Hand Gesture Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    # Timing variables for FPS computation
    prev_time = time.time()
    fps_smooth = 30.0

    print("\n[READY] Starting video stream processing. Enjoy gesture control!")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror the frame horizontally if configured
            if config.MIRROR_FEED:
                frame = cv2.flip(frame, 1)

            # Compute smoothed FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            # Find hands in the frame
            hands_landmarks = detector.find_hands(frame)
            gesture_info = None
            hand_detected = bool(hands_landmarks)

            if hand_detected:
                # Use dominant/primary hand
                primary_hand = hands_landmarks[0]
                detector.draw_landmarks(frame, primary_hand)
                gesture_info = detector.classify_gesture(primary_hand)
                controller.process_gesture(gesture_info)
            else:
                detector.clear_motion_history()
                controller.process_gesture(None)

            # Render modern Cyber HUD overlay
            display_frame = ui_overlay.draw_hud(
                frame,
                fps=fps_smooth,
                gesture_info=gesture_info,
                controller=controller,
                hand_detected=hand_detected
            )

            cv2.imshow(window_name, display_frame)

            # Check for keyboard inputs on the OpenCV window
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
                print("[INFO] Quit requested by user.")
                break
            elif key == ord(' '):  # Spacebar toggles control pause
                controller.toggle_active()

            # Check if window was closed by the user clicking 'X'
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Control window closed by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down gracefully...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Windows closed. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
