#!/usr/bin/env python3
"""
Darts Tracker v0.2
- カメラ自動検出
- ダーツ座標・スコア記録
- 軌跡・リリースポイント判定
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path

# ─── データモデル ────────────────────────────────────────
@dataclass
class DartThrow:
    timestamp: str
    board_x: float          # ボード座標 (-1.0〜1.0, 中心=0)
    board_y: float
    score: int
    zone: str               # "single", "double", "treble", "bull", "outer_bull"
    release_point: tuple    # (x, y) ピクセル座標
    trajectory_points: list # 軌跡ポイントリスト

# ─── ダーツボードスコアリング ─────────────────────────────
SEGMENTS = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

def coords_to_score(nx: float, ny: float) -> tuple[int, str]:
    """正規化座標(-1〜1) → (スコア, ゾーン名)"""
    r = np.sqrt(nx**2 + ny**2)
    if r < 0.05:   return 50, "bull"
    if r < 0.12:   return 25, "outer_bull"
    angle = (np.degrees(np.arctan2(-nx, -ny)) + 360) % 360
    seg_idx = int(angle / 18) % 20
    number = SEGMENTS[seg_idx]
    if r < 0.55:   return number * 3, "treble"
    if r < 0.75:   return number,     "single"
    if r < 0.95:   return number * 2, "double"
    return 0, "miss"

# ─── ボード検出・キャリブレーション ──────────────────────
class BoardCalibrator:
    def __init__(self):
        self.center = None
        self.radius = None
        self.calibrated = False

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                                   param1=50, param2=30, minRadius=50, maxRadius=400)
        if circles is not None:
            c = np.round(circles[0][0]).astype(int)
            self.center = (c[0], c[1])
            self.radius = c[2]
            self.calibrated = True
            return True
        return False

    def to_board_coords(self, px, py):
        if not self.calibrated:
            return 0.0, 0.0
        nx = (px - self.center[0]) / self.radius
        ny = (py - self.center[1]) / self.radius
        return nx, ny

    def draw(self, frame):
        if self.calibrated:
            cv2.circle(frame, self.center, self.radius, (0, 255, 0), 2)
            cv2.circle(frame, self.center, 5, (0, 0, 255), -1)
            # ゾーン円
            for ratio, label in [(0.05,'B'),(0.12,'OB'),(0.55,'T'),(0.75,'S'),(0.95,'D')]:
                cv2.circle(frame, self.center, int(self.radius*ratio), (100,100,255), 1)

# ─── ダーツ追跡 ──────────────────────────────────────────
class DartTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False)
        self.trail = deque(maxlen=60)
        self.state = "idle"  # idle → in_flight → landed
        self.release_point = None
        self.throws = []
        self.idle_frames = 0

    def process(self, frame, calibrator):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg_sub.apply(blur)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 0.1)
            if aspect > 2.0:  # 細長い→ダーツらしい
                detected = (x + w//2, y + h//2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
                break

        # 状態機械
        if detected:
            self.idle_frames = 0
            if self.state == "idle":
                self.state = "in_flight"
                self.release_point = detected  # 最初の検出 ≈ リリース後
            self.trail.append(detected)
        else:
            self.idle_frames += 1
            if self.state == "in_flight" and self.idle_frames > 10:
                # 静止 → 刺さった
                if len(self.trail) >= 3:
                    landing = self.trail[-1]
                    nx, ny = calibrator.to_board_coords(*landing)
                    score, zone = coords_to_score(nx, ny)
                    throw = DartThrow(
                        timestamp=datetime.now().isoformat(),
                        board_x=round(nx, 3),
                        board_y=round(ny, 3),
                        score=score,
                        zone=zone,
                        release_point=self.release_point,
                        trajectory_points=list(self.trail),
                    )
                    self.throws.append(throw)
                    self._save(throw)
                    print(f"[THROW] score={score} zone={zone} pos=({nx:.2f},{ny:.2f})")
                self.trail.clear()
                self.release_point = None
                self.state = "idle"

        return detected, mask

    def _save(self, throw: DartThrow):
        path = Path("throws.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(asdict(throw), ensure_ascii=False) + "\n")

    def draw(self, frame):
        # 軌跡
        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (0, int(255*alpha), int(255*(1-alpha)))
            cv2.line(frame, pts[i-1], pts[i], color, 2)
        if pts:
            cv2.circle(frame, pts[-1], 8, (0, 0, 255), -1)
        # リリースポイント
        if self.release_point:
            cv2.circle(frame, self.release_point, 12, (255, 165, 0), 2)
            cv2.putText(frame, "Release", (self.release_point[0]+10, self.release_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)
        # 状態
        color = {"idle":(128,128,128),"in_flight":(0,255,0),"landed":(0,0,255)}.get(self.state,(255,255,255))
        cv2.putText(frame, f"State: {self.state}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ─── メインループ ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: カメラが開けませんでした")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"カメラ起動: {int(cap.get(3))}x{int(cap.get(4))} @ {fps}fps")

    calibrator = BoardCalibrator()
    tracker = DartTracker(fps=fps)
    total_score = 0
    throw_count = 0

    print("操作: 'c'=ボード検出 | 'r'=リセット | 'q'=終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ボード自動検出（未キャリブレーション時）
        if not calibrator.calibrated and throw_count == 0:
            calibrator.detect(frame)

        calibrator.draw(frame)
        detected, mask = tracker.process(frame, calibrator)
        tracker.draw(frame)

        # 新スロー検出
        if len(tracker.throws) > throw_count:
            throw = tracker.throws[-1]
            throw_count = len(tracker.throws)
            total_score += throw.score
            cv2.putText(frame, f"+{throw.score} ({throw.zone})", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # HUD
        cv2.putText(frame, f"Total: {total_score}  Throws: {throw_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Board: {'OK' if calibrator.calibrated else 'Press C'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if calibrator.calibrated else (0,100,255), 2)
        cv2.putText(frame, f"Saved: throws.jsonl", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        cv2.imshow("Darts Tracker", frame)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if calibrator.detect(frame):
                print(f"ボード検出: center={calibrator.center} r={calibrator.radius}")
            else:
                print("ボード未検出。もう少しはっきり映してくださいわ")
        elif key == ord('r'):
            tracker = DartTracker(fps=fps)
            total_score = 0
            throw_count = 0
            print("リセットしましたわ")

    cap.release()
    cv2.destroyAllWindows()
    print(f"終了: {throw_count}投 合計{total_score}点  → throws.jsonl に保存済み")

if __name__ == "__main__":
    main()
