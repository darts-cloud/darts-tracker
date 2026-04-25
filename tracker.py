#!/usr/bin/env python3
"""
Darts Tracker - 起動確認版
20km/h (5.56m/s) @ 30fps: ~18cm/frame 追跡対応
"""

import cv2
import numpy as np
import time
from collections import deque

# 設定
CAMERA_INDEX = 0
TARGET_FPS = 60
DART_SPEED_KMH = 20
TRAIL_LENGTH = 15  # 軌跡の長さ（フレーム数）

def estimate_dart_speed(positions, fps):
    """ピクセル移動量から速度を推定（px/s）"""
    if len(positions) < 2:
        return 0
    p1, p2 = positions[-2], positions[-1]
    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    return dist * fps

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした。CAMERA_INDEX を確認してください。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラ起動: {width}x{height} @ {actual_fps}fps")

    # 背景差分（MOG2: 照明変化に強い）
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=False
    )

    trail = deque(maxlen=TRAIL_LENGTH)
    frame_count = 0
    fps_timer = time.time()
    display_fps = 0

    print("'q'で終了 | 'r'でリセット | 起動確認OK")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            display_fps = 30 / elapsed
            fps_timer = time.time()

        # 前処理
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = bg_sub.apply(blur)

        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 輪郭検出
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = None
        max_area = 200  # 最小面積フィルタ

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                # ダーツらしい細長い形状を優先
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1)
                if aspect > 1.5:  # 細長い物体
                    cx, cy = x + w // 2, y + h // 2
                    detected = (cx, cy)
                    max_area = area
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 軌跡記録・描画
        if detected:
            trail.append(detected)

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, trail[i-1], trail[i], color, 2)

        if trail:
            cv2.circle(frame, trail[-1], 8, (0, 0, 255), -1)
            speed_px = estimate_dart_speed(trail, actual_fps)
            cv2.putText(frame, f"Speed: {speed_px:.0f}px/s",
                        (trail[-1][0]+10, trail[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # HUD
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Target: {DART_SPEED_KMH}km/h @ {TARGET_FPS}fps",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Objects: {len(contours)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Darts Tracker - 起動確認版", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            trail.clear()
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=50, detectShadows=False
            )
            print("リセットしましたわ")

    cap.release()
    cv2.destroyAllWindows()
    print("終了しましたわ")

if __name__ == "__main__":
    main()
