"""
Automated Test and Diagnostic Suite for YouTube Hand Gesture Controller.
Validates dependencies, model initialization, gesture classifier logic,
and PyAutoGUI debounce/cooldown mechanics without requiring an active webcam.
"""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from gesture_detector import GestureDetector
from youtube_controller import YouTubeController
import ui_overlay


def run_diagnostics():
    print("=" * 60)
    print("  RUNNING YOUTUBE GESTURE CONTROLLER DIAGNOSTIC SUITE")
    print("=" * 60)
    passed_tests = 0
    total_tests = 5

    # -------------------------------------------------------------
    # Test 1: Configuration and Model File Verification
    # -------------------------------------------------------------
    print("\n[TEST 1/5] Checking Configuration and Model File...")
    assert os.path.exists(config.MODEL_PATH), f"Model file not found at {config.MODEL_PATH}"
    model_size = os.path.getsize(config.MODEL_PATH)
    print(f" -> Model exists at: {config.MODEL_PATH} ({model_size / (1024*1024):.2f} MB)")
    assert model_size > 1000000, "Model file seems too small or corrupt!"
    print(" -> [PASS] Test 1: Model file valid.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 2: GestureDetector Initialization & Inference Test
    # -------------------------------------------------------------
    print("\n[TEST 2/5] Initializing GestureDetector & testing dummy frame...")
    detector = GestureDetector()
    assert detector is not None
    print(f" -> MediaPipe engine mode: {'Tasks API' if detector.use_tasks_api else 'Legacy Solutions'}")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = detector.find_hands(dummy_frame)
    print(f" -> Inference on blank frame returned {len(hands)} hands (expected 0).")
    assert isinstance(hands, list)
    print(" -> [PASS] Test 2: GestureDetector initialized and processed frame without error.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 3: Finger States and Gesture Classification Logic
    # -------------------------------------------------------------
    print("\n[TEST 3/5] Testing Geometric Landmark Logic with Mock Landmarks...")
    # Create mock 21 landmarks representing an Open Palm (all extended)
    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    # Upright open palm mock
    mock_open_palm = [MockLandmark(0.5, 0.8)]  # 0: wrist
    # Thumb (1, 2, 3, 4)
    mock_open_palm += [MockLandmark(0.42, 0.7), MockLandmark(0.38, 0.65), MockLandmark(0.32, 0.6), MockLandmark(0.25, 0.55)]
    # Index (5, 6, 7, 8)
    mock_open_palm += [MockLandmark(0.45, 0.5), MockLandmark(0.45, 0.4), MockLandmark(0.45, 0.3), MockLandmark(0.45, 0.2)]
    # Middle (9, 10, 11, 12)
    mock_open_palm += [MockLandmark(0.50, 0.48), MockLandmark(0.50, 0.38), MockLandmark(0.50, 0.28), MockLandmark(0.50, 0.18)]
    # Ring (13, 14, 15, 16)
    mock_open_palm += [MockLandmark(0.55, 0.5), MockLandmark(0.55, 0.4), MockLandmark(0.55, 0.3), MockLandmark(0.55, 0.22)]
    # Pinky (17, 18, 19, 20)
    mock_open_palm += [MockLandmark(0.60, 0.55), MockLandmark(0.60, 0.46), MockLandmark(0.60, 0.38), MockLandmark(0.60, 0.30)]

    finger_states = detector.get_finger_states(mock_open_palm)
    print(f" -> Mock Open Palm finger states [Thumb, Index, Mid, Ring, Pinky]: {finger_states}")
    classification = detector.classify_gesture(mock_open_palm)
    print(f" -> Open Palm result: {classification['gesture']} -> Action: {classification['action']}")
    assert classification['action'] == "PLAY_PAUSE", f"Expected PLAY_PAUSE, got {classification['action']}"

    # Mock Closed Fist (all tips near wrist / MCP)
    mock_fist = [MockLandmark(0.5, 0.8)]  # wrist
    # Thumb curled
    mock_fist += [MockLandmark(0.48, 0.75), MockLandmark(0.48, 0.72), MockLandmark(0.48, 0.70), MockLandmark(0.48, 0.68)]
    # Index curled
    mock_fist += [MockLandmark(0.46, 0.65), MockLandmark(0.46, 0.68), MockLandmark(0.46, 0.70), MockLandmark(0.46, 0.72)]
    # Middle curled
    mock_fist += [MockLandmark(0.50, 0.65), MockLandmark(0.50, 0.68), MockLandmark(0.50, 0.70), MockLandmark(0.50, 0.72)]
    # Ring curled
    mock_fist += [MockLandmark(0.54, 0.65), MockLandmark(0.54, 0.68), MockLandmark(0.54, 0.70), MockLandmark(0.54, 0.72)]
    # Pinky curled
    mock_fist += [MockLandmark(0.58, 0.65), MockLandmark(0.58, 0.68), MockLandmark(0.58, 0.70), MockLandmark(0.58, 0.72)]

    fist_classification = detector.classify_gesture(mock_fist)
    print(f" -> Closed Fist result: {fist_classification['gesture']} -> Action: {fist_classification['action']}")
    assert fist_classification['action'] == "MUTE_UNMUTE", f"Expected MUTE_UNMUTE, got {fist_classification['action']}"

    # Mock Two Fingers (Index + Middle extended, others curled)
    mock_peace = list(mock_fist)
    # Extend Index (5, 6, 7, 8)
    mock_peace[5:9] = [MockLandmark(0.46, 0.5), MockLandmark(0.46, 0.4), MockLandmark(0.46, 0.3), MockLandmark(0.46, 0.2)]
    # Extend Middle (9, 10, 11, 12)
    mock_peace[9:13] = [MockLandmark(0.50, 0.48), MockLandmark(0.50, 0.38), MockLandmark(0.50, 0.28), MockLandmark(0.50, 0.18)]
    peace_classification = detector.classify_gesture(mock_peace)
    print(f" -> Peace Sign result: {peace_classification['gesture']} -> Action: {peace_classification['action']}")
    assert peace_classification['action'] == "FULLSCREEN", f"Expected FULLSCREEN, got {peace_classification['action']}"

    print(" -> [PASS] Test 3: Gesture classification logic verified for all major poses.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 4: YouTube Controller Debounce & Cooldown Engine
    # -------------------------------------------------------------
    print("\n[TEST 4/5] Testing YouTubeController Debounce and Cooldown...")
    controller = YouTubeController()

    # Disable actual key sending for dry-run testing
    executed_actions = []
    original_execute = controller._execute_action
    controller._execute_action = lambda action_name, gesture_name: executed_actions.append(action_name) or (action_name, "TEST_OK")

    test_gesture = {"gesture": "OPEN_PALM", "action": "PLAY_PAUSE"}

    # Frames 1 to 3 should not trigger (requires GESTURE_CONFIRM_FRAMES = 4)
    for i in range(1, config.GESTURE_CONFIRM_FRAMES):
        act, _ = controller.process_gesture(test_gesture)
        assert act is None, f"Triggered prematurely on frame {i}!"

    # Frame 4 should trigger!
    act, _ = controller.process_gesture(test_gesture)
    assert act == "PLAY_PAUSE", f"Failed to trigger on frame {config.GESTURE_CONFIRM_FRAMES}!"
    assert len(executed_actions) == 1, "Expected exactly 1 execution!"
    print(" -> Confirmed multi-frame debounce logic works properly.")

    # Immediately sending same gesture should be blocked by cooldown
    act, _ = controller.process_gesture(test_gesture)
    assert act is None, "Failed to block immediate repeat action under cooldown!"
    print(f" -> Cooldown properly active ({controller.get_cooldown_remaining('PLAY_PAUSE'):.2f}s remaining).")
    print(" -> [PASS] Test 4: Debounce and cooldown engine verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 5: UI Overlay Rendering Test
    # -------------------------------------------------------------
    print("\n[TEST 5/5] Testing UI Overlay HUD rendering...")
    test_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered_canvas = ui_overlay.draw_hud(
        test_canvas,
        fps=30.0,
        gesture_info=test_gesture,
        controller=controller,
        hand_detected=True
    )
    assert rendered_canvas.shape == (720, 1280, 3)
    assert np.mean(rendered_canvas) > 0, "HUD did not render any pixels!"
    print(" -> [PASS] Test 5: Cyber HUD rendered cleanly on 1280x720 frame.")
    passed_tests += 1

    print("\n" + "=" * 60)
    print(f"  ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! SYSTEM READY.")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
