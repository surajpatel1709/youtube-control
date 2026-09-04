import time
import threading
from collections import deque, Counter
import pyautogui
import config

# Configure PyAutoGUI for zero latency
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class YouTubeController:
    """
    Executes YouTube web player commands via keyboard simulation with
    multi-frame confirmation and per-action cooldown safety.
    """

    def __init__(self):
        self.is_active = True
        
        # Rolling history buffer of recent detected gestures for jitter-free majority voting
        self.gesture_history = deque(maxlen=config.GESTURE_CONFIRM_FRAMES + 2)
        
        # Timestamp records for cooldown calculation
        self.last_action_times = {action: 0.0 for action in config.COOLDOWNS}
        
        # Recent action tracking for UI presentation
        self.last_triggered_action = "SYSTEM READY"
        self.last_triggered_time = time.time()
        self.last_triggered_gesture = "NONE"

    def toggle_active(self):
        """Toggles gesture control state between ACTIVE and PAUSED/SLEEP."""
        self.is_active = not self.is_active
        state_str = "ACTIVE" if self.is_active else "PAUSED"
        self.last_triggered_action = f"CONTROL {state_str}"
        self.last_triggered_time = time.time()
        print(f"[STATUS] Gesture Control is now: {state_str}")

    def get_cooldown_remaining(self, action):
        """Returns the remaining cooldown time in seconds for a specific action."""
        if action not in config.COOLDOWNS:
            return 0.0
        elapsed = time.time() - self.last_action_times.get(action, 0.0)
        cooldown = config.COOLDOWNS[action]
        return max(0.0, cooldown - elapsed)

    def is_action_ready(self, action):
        """Checks if enough time has elapsed since the action was last triggered."""
        return self.get_cooldown_remaining(action) <= 0.0

    def process_gesture(self, gesture_info):
        """
        Evaluates a detected gesture from the detector.
        Applies rolling window confirmation and cooldown checks before executing.
        
        Returns:
            tuple: (action_executed_or_none, feedback_message)
        """
        if not gesture_info:
            self.gesture_history.append("NONE")
            return None, None

        gesture_name = gesture_info.get("gesture")
        action_name = gesture_info.get("action")

        if not action_name:
            self.gesture_history.append("NONE")
            return None, None

        now = time.time()

        # Handle TOGGLE_ACTIVE regardless of current active state
        if action_name == "TOGGLE_ACTIVE":
            if self.is_action_ready("TOGGLE_ACTIVE"):
                self.gesture_history.append(gesture_name)
                # Check if gesture appears in at least 2 of recent frames
                if self.gesture_history.count(gesture_name) >= 2:
                    self.last_action_times["TOGGLE_ACTIVE"] = now
                    self.toggle_active()
                    self.gesture_history.clear()
                    return "TOGGLE_ACTIVE", self.last_triggered_action
            return None, None

        # If control is paused/disabled, ignore all playback actions
        if not self.is_active:
            return None, "GESTURE CONTROL PAUSED (Show Call-Me gesture or press Space to wake)"

        # Check action cooldown
        if not self.is_action_ready(action_name):
            return None, f"COOLDOWN ({action_name})"

        # Dynamic / Swipe gestures already span multiple frames, trigger immediately
        is_swipe = "SWIPE" in gesture_name or "MOVE" in gesture_name
        if is_swipe:
            self.gesture_history.clear()
            return self._execute_action(action_name, gesture_name)

        # Static gestures: append to rolling history
        self.gesture_history.append(gesture_name)

        # Check majority vote in recent history to eliminate frame-to-frame flicker
        gesture_counts = Counter(self.gesture_history)
        if gesture_counts[gesture_name] >= config.GESTURE_CONFIRM_FRAMES:
            self.gesture_history.clear()
            return self._execute_action(action_name, gesture_name)

        return None, None

    def _execute_action(self, action_name, gesture_name):
        """Dispatches the corresponding keyboard shortcut asynchronously via PyAutoGUI."""
        hotkey = config.HOTKEYS.get(action_name)
        if not hotkey:
            return None, None

        now = time.time()
        self.last_action_times[action_name] = now
        self.last_triggered_action = action_name.replace("_", " ")
        self.last_triggered_time = now
        self.last_triggered_gesture = gesture_name

        def _press():
            try:
                if isinstance(hotkey, (list, tuple)):
                    pyautogui.hotkey(*hotkey)
                else:
                    pyautogui.press(hotkey)
            except Exception as err:
                print(f"[ERROR] Hotkey dispatch failed: {err}")

        # Run keystroke in background thread to guarantee 0ms lag on video frame loop
        threading.Thread(target=_press, daemon=True).start()

        key_label = " + ".join(hotkey).upper() if isinstance(hotkey, (list, tuple)) else str(hotkey).upper()
        log_msg = f"[ACTION] Triggered '{action_name}' via '{gesture_name}' [Key: {key_label}]"
        print(log_msg)
        return action_name, log_msg
