#!/usr/bin/env python3
"""
Darts Player Tracker
- プレイヤー側カメラで投げたダーツを追跡
- リリースポイント・軌跡・速度を記録
- ボード認識なし（スコア入力は別途）
"""

import cv2
import numpy as np
import json
from datetime import datetime
from collections import deque
from pathlib import Path

# ─── 設定 ─────────────────────────────────────────────────
CAMERA_INDEX = 0
TARGET_FPS   = 60
TRAIL_LEN    = 90   # 軌跡保持フレーム数（60fps × 1.5秒分）
MIN_AREA     = 80   # 検出最小面積 (px²)
MIN_ASPECT   = 2.0  # ダーツらしい縦横比（細長さ）
IDLE_FRAMES  = 20   # N フレーム動体なし → 投擲終了と判定

# ─── データ保存 ───────────────────────────────────────────
LOG_FILE = Path("throws.jsonl")

def save_throw(release_pt, trail, fps):
    if len(trail) < 3:
        return
    pts = list(trail)
    # 速度推定（最初の数フレームの移動量から）
    dists = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
             for i in range(1, min(6, len(pts)))]
    speed_px_s = np.mean(dists) * fps if dists else 0
    record = {
        "timestamp":     datetime.now().isoformat(),
        "release_point": list(release_pt),
        "landing_point": list(pts[-1]),
        "speed_px_s":    round(speed_px_s, 1),
        "frame_count":   len(pts),
        "trajectory":    pts,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[SAVED] release={release_pt}  frames={len(pts)}  speed={speed_px_s:.0f}px/s")
    return record

# ─── 追跡器 ──────────────────────────────────────────────
class PlayerTracker:
    def __init__(self, fps):
        self.fps   = fps
        self.bg    = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=40, detectShadows=False)
        self.trail       = deque(maxlen=TRAIL_LEN)
        self.release_pt  = None
        self.in_flight   = False
        self.idle        = 0
        self.last_saved  = None

    def _detect(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(blur)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            asp = max(w, h) / (min(w, h) + 0.1)
            if asp >= MIN_ASPECT and area > best_score:
                best       = (x + w // 2, y + h // 2, x, y, w, h)
                best_score = area
        return best, mask

    def process(self, frame):
        det, mask = self._detect(frame)
        saved = None

        if det:
            cx, cy, x, y, w, h = det
            self.idle = 0
            if not self.in_flight:
                self.in_flight  = True
                self.release_pt = (cx, cy)
                self.trail.clear()
            self.trail.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 80), 2)
        else:
            if self.in_flight:
                self.idle += 1
                if self.idle >= IDLE_FRAMES:
                    saved = save_throw(self.release_pt, self.trail, self.fps)
                    self.last_saved = saved
                    self.in_flight  = False
                    self.idle       = 0

        # 軌跡描画
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(frame, pts[i-1], pts[i],
                     (0, int(255*a), int(255*(1-a))), 2)
        if pts:
            cv2.circle(frame, pts[-1], 7, (0, 0, 255), -1)

        # リリースポイント
        if self.release_pt:
            cv2.circle(frame, self.release_pt, 14, (255, 140, 0), 2)
            cv2.putText(frame, "Release", (self.release_pt[0]+8, self.release_pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 1)

        return mask, saved

# ─── メイン ──────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    fps    = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラ起動: {width}×{height} @ {fps:.0f}fps")
    print(f"操作: 'r'=リセット | 'q'=終了  →  記録: {LOG_FILE}")

    tracker     = PlayerTracker(fps)
    throw_count = 0
    overlay_msg = ""
    overlay_ttl = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, saved = tracker.process(frame)

        if saved:
            throw_count += 1
            overlay_msg = (f"#{throw_count}  speed={saved['speed_px_s']:.0f}px/s"
                           f"  frames={saved['frame_count']}")
            overlay_ttl = int(fps * 2)  # 2秒表示

        # HUD
        state_col = (0, 255, 0) if tracker.in_flight else (160, 160, 160)
        cv2.putText(frame, f"{'IN FLIGHT' if tracker.in_flight else 'WAITING'}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_col, 2)
        cv2.putText(frame, f"Throws: {throw_count}  |  {fps:.0f}fps",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 0), 2)
        cv2.putText(frame, f"Log: {LOG_FILE}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        if overlay_ttl > 0:
            cv2.putText(frame, overlay_msg, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            overlay_ttl -= 1

        cv2.imshow("Darts Player Tracker", frame)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker = PlayerTracker(fps)
            throw_count = 0
            print("リセットしましたわ")

    cap.release()
    cv2.destroyAllWindows()
    print(f"終了: {throw_count}投  →  {LOG_FILE}")

if __name__ == "__main__":
    main()
