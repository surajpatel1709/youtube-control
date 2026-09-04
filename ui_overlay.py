"""
UI Overlay Module for YouTube Hand Gesture Controller.
Renders a modern, translucent Cyber HUD over the webcam feed,
displaying FPS, gesture detection badge, action flash banner,
cooldown progress bar, and an on-screen gesture cheat sheet.
"""

import time
import cv2
import numpy as np
import config


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draws a rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.rectangle(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.rectangle(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_hud(frame, fps, gesture_info, controller, hand_detected=False):
    """
    Overlays status information, current gesture, action triggered,
    live finger states, and gesture reference guide on the video frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # ==========================================
    # 1. TOP STATUS BAR (Translucent Banner)
    # ==========================================
    bar_height = 48
    cv2.rectangle(overlay, (0, 0), (w, bar_height), config.COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # App Title
    cv2.putText(frame, "YT GESTURE CONTROL", (14, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, config.COLOR_PRIMARY, 2, cv2.LINE_AA)

    # Status Pill (ACTIVE vs PAUSED)
    status_str = "ACTIVE" if controller.is_active else "PAUSED"
    status_color = config.COLOR_SUCCESS if controller.is_active else config.COLOR_DANGER
    cv2.putText(frame, f"STATUS: {status_str}", (w - 290, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    # FPS Display
    fps_color = config.COLOR_SUCCESS if fps >= 22 else config.COLOR_WARNING
    cv2.putText(frame, f"FPS: {int(fps):02d}", (w - 130, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    # Hand Presence Dot
    hand_color = config.COLOR_SUCCESS if hand_detected else config.COLOR_TEXT_MUTED
    cv2.circle(frame, (w - 20, 28), 6, hand_color, -1, cv2.LINE_AA)

    # ==========================================
    # 2. MAIN HUD CARD (Top Left: Gesture, Action & Fingers)
    # ==========================================
    card_overlay = frame.copy()
    card_x, card_y = 12, 58
    card_w = min(360, w - 24)
    card_h = 160
    draw_rounded_rect(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      config.COLOR_BG_DARK, radius=10, thickness=-1)
    cv2.addWeighted(card_overlay, 0.82, frame, 0.18, 0, frame)

    # Card Border
    draw_rounded_rect(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      (60, 75, 90), radius=10, thickness=1)

    # Formatted Gesture & Action Text
    detected_gesture = gesture_info.get("gesture", "SEARCHING...") if gesture_info else "SEARCHING..."
    curr_action = gesture_info.get("action", "NONE") if gesture_info else "NONE"
    if curr_action is None:
        curr_action = "NONE"

    disp_gesture = detected_gesture.replace("_", " ").title()
    disp_action = curr_action.replace("_", " ").upper()

    cv2.putText(frame, "GESTURE:", (card_x + 14, card_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, disp_gesture, (card_x + 108, card_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_PRIMARY, 2, cv2.LINE_AA)

    cv2.putText(frame, "ACTION:", (card_x + 14, card_y + 66),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    action_color = config.COLOR_SUCCESS if disp_action != "NONE" else config.COLOR_TEXT_MUTED
    cv2.putText(frame, disp_action, (card_x + 108, card_y + 66),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, action_color, 2, cv2.LINE_AA)

    # Live Finger State Badges [T, I, M, R, P]
    cv2.putText(frame, "FINGERS:", (card_x + 14, card_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    finger_labels = ["T", "I", "M", "R", "P"]
    finger_states = gesture_info.get("finger_states", [False]*5) if gesture_info else [False]*5
    for idx, (label, state) in enumerate(zip(finger_labels, finger_states)):
        bx = card_x + 110 + (idx * 38)
        by = card_y + 90
        b_color = config.COLOR_SUCCESS if state else (50, 50, 70)
        cv2.rectangle(frame, (bx, by), (bx + 28, by + 20), b_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + 28, by + 20), (140, 140, 150), 1)
        text_col = (20, 20, 20) if state else config.COLOR_TEXT_MUTED
        cv2.putText(frame, label, (bx + 8, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1, cv2.LINE_AA)

    # Cooldown Status Line
    cooldown_action = gesture_info.get("action") if gesture_info else None
    remaining_cd = controller.get_cooldown_remaining(cooldown_action) if cooldown_action else 0.0
    if remaining_cd > 0:
        cd_text = f"COOLDOWN: {remaining_cd:.1f}s"
        cv2.putText(frame, cd_text, (card_x + 14, card_y + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, config.COLOR_WARNING, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: READY", (card_x + 14, card_y + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, config.COLOR_SUCCESS, 1, cv2.LINE_AA)

    # ==========================================
    # 3. ACTION FLASH BANNER (Animated Toast)
    # ==========================================
    time_since_trigger = time.time() - controller.last_triggered_time
    if time_since_trigger < 1.3:
        banner_overlay = frame.copy()
        banner_w, banner_h = min(460, w - 40), 44
        bx1 = (w - banner_w) // 2
        by1 = h - 60
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        alpha = max(0.25, 0.85 * (1.0 - time_since_trigger / 1.3))
        draw_rounded_rect(banner_overlay, (bx1, by1), (bx2, by2), config.COLOR_SUCCESS, radius=8, thickness=-1)
        cv2.addWeighted(banner_overlay, alpha, frame, 1.0 - alpha, 0, frame)
        draw_rounded_rect(frame, (bx1, by1), (bx2, by2), (255, 255, 255), radius=8, thickness=2)

        banner_text = f">> {controller.last_triggered_action} <<"
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
        tx = bx1 + (banner_w - text_size[0]) // 2
        ty = by1 + (banner_h + text_size[1]) // 2
        cv2.putText(frame, banner_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    # ==========================================
    # 4. GESTURE CHEAT SHEET (Right Panel, if space permits)
    # ==========================================
    if w >= 600:
        guide_w, guide_h = 240, 200
        gx1 = w - guide_w - 12
        gy1 = 58
        gx2 = gx1 + guide_w
        gy2 = gy1 + guide_h

        guide_overlay = frame.copy()
        draw_rounded_rect(guide_overlay, (gx1, gy1), (gx2, gy2), config.COLOR_BG_DARK, radius=10, thickness=-1)
        cv2.addWeighted(guide_overlay, 0.82, frame, 0.18, 0, frame)
        draw_rounded_rect(frame, (gx1, gy1), (gx2, gy2), (60, 75, 90), radius=10, thickness=1)

        cv2.putText(frame, "QUICK GUIDE", (gx1 + 12, gy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, config.COLOR_PRIMARY, 2, cv2.LINE_AA)

        cheatsheet = [
            ("Palm Hold", "Play / Pause"),
            ("Palm Swipe L/R", "Next / Prev"),
            ("Point Swipe L/R", "Seek 10s"),
            ("Point Up / Down", "Volume +/-"),
            ("Closed Fist", "Mute / Unmute"),
            ("Victory (✌)", "Fullscreen"),
            ("Call-Me (🤙)", "Sleep Toggle"),
        ]

        for i, (gest, act) in enumerate(cheatsheet):
            item_y = gy1 + 44 + (i * 22)
            cv2.putText(frame, gest, (gx1 + 10, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, act, (gx1 + 125, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_SUCCESS, 1, cv2.LINE_AA)

    return frame
