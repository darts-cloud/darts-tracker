#!/usr/bin/env python3
"""
Darts Board Camera
- ボード正面カメラでダーツの着弾座標を記録
- ゾーン判定・スコア計算なし（座標のみ）
- 飛行追跡なし
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ─── 設定 ─────────────────────────────────────────────────
CAMERA_INDEX = 0
TARGET_FPS   = 60
LOG_FILE     = Path("landings.jsonl")

# ─── ボード基準点（手動キャリブレーション） ───────────────
class BoardRef:
    """ボード中心と半径を手動指定して正規化座標に変換"""
    def __init__(self):
        self.center = None
        self.radius = None

    @property
    def ready(self):
        return self.center is not None and self.radius is not None

    def normalize(self, px, py):
        """ピクセル座標 → ボード正規化座標 (-1〜1)"""
        nx = (px - self.center[0]) / self.radius
        ny = (py - self.center[1]) / self.radius
        return round(nx, 4), round(ny, 4)

    def draw(self, frame):
        if not self.ready:
            cv2.putText(frame, "Press S: set center  R: set radius",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)
            return
        cv2.circle(frame, self.center, self.radius, (0, 200, 0), 2)
        cv2.circle(frame, self.center, 5, (0, 0, 255), -1)

# ─── ダーツ着弾検出 ──────────────────────────────────────
class LandingDetector:
    """
    前フレームとの差分が大きく、その後静止したら着弾と判定。
    ボードに刺さったダーツの軸・フライトを細長い輪郭として検出。
    """
    def __init__(self):
        self.bg       = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=30, detectShadows=False)
        self.stable   = 0          # 静止フレーム数
        self.pending  = None       # 着弾候補座標
        self.STABLE_N = 12         # N フレーム静止で確定

    def process(self, frame):
        blur  = cv2.GaussianBlur(frame, (5, 5), 0)
        mask  = self.bg.apply(blur)
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)

        confirmed = None

        if motion_area > 200:
            # 動きあり：着弾候補を更新
            self.stable = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 60:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                asp = max(w, h) / (min(w, h) + 0.1)
                if asp >= 2.5:   # ダーツ軸らしい細長さ
                    self.pending = (x + w // 2, y + h // 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
        else:
            # 静止中
            if self.pending:
                self.stable += 1
                if self.stable >= self.STABLE_N:
                    confirmed      = self.pending
                    self.pending   = None
                    self.stable    = 0

        return mask, confirmed

# ─── 保存 ─────────────────────────────────────────────────
def save_landing(px, py, board_ref, throw_n):
    entry = {
        "timestamp":  datetime.now().isoformat(),
        "throw_n":    throw_n,
        "pixel":      [px, py],
    }
    if board_ref.ready:
        nx, ny = board_ref.normalize(px, py)
        entry["board_norm"] = [nx, ny]
        entry["radius_norm"] = round(np.hypot(nx, ny), 4)

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[LANDING #{throw_n}] pixel=({px},{py})"
          + (f"  norm=({entry['board_norm'][0]},{entry['board_norm'][1]})"
             if board_ref.ready else ""))
    return entry

# ─── メイン ──────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    fps    = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラ: {width}×{height} @ {fps:.0f}fps")

    ref      = BoardRef()
    detector = LandingDetector()
    throws   = 0
    last_pt  = None
    click_mode = None  # "center" or "radius"

    def on_mouse(event, x, y, flags, param):
        nonlocal click_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_mode == "center":
                ref.center = (x, y)
                print(f"中心設定: {ref.center}")
                click_mode = None
            elif click_mode == "radius":
                if ref.center:
                    ref.radius = int(np.hypot(x - ref.center[0], y - ref.center[1]))
                    print(f"半径設定: {ref.radius}px")
                click_mode = None

    cv2.namedWindow("Board Camera")
    cv2.setMouseCallback("Board Camera", on_mouse)

    print("操作: S=中心クリック設定  E=半径クリック設定  R=リセット  Q=終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, confirmed = detector.process(frame)
        ref.draw(frame)

        if confirmed:
            throws += 1
            last_pt = confirmed
            save_landing(*confirmed, ref, throws)
            # 着弾マーカー
        if last_pt:
            cv2.circle(frame, last_pt, 10, (0, 0, 255), -1)
            cv2.circle(frame, last_pt, 10, (255, 255, 255), 2)
            label = f"#{throws}"
            if ref.ready:
                nx, ny = ref.normalize(*last_pt)
                label += f"  ({nx:+.3f}, {ny:+.3f})"
            cv2.putText(frame, label,
                        (last_pt[0]+12, last_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # HUD
        cv2.putText(frame, f"Throws: {throws}  |  {fps:.0f}fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 0), 2)
        ref_status = f"Ref: center={ref.center} r={ref.radius}" if ref.ready else "Ref: NOT SET (S=center E=radius)"
        cv2.putText(frame, ref_status,
                    (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        if click_mode:
            cv2.putText(frame, f">>> クリックしてください: {click_mode} <<<",
                        (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)

        cv2.imshow("Board Camera", frame)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            click_mode = "center"
            print("画面上でボードの中心をクリックしてくださいわ")
        elif key == ord('e'):
            click_mode = "radius"
            print("画面上でボードの端（外輪）をクリックしてくださいわ")
        elif key == ord('r'):
            detector = LandingDetector()
            throws   = 0
            last_pt  = None
            print("リセットしましたわ")

    cap.release()
    cv2.destroyAllWindows()
    print(f"終了: {throws}投  →  {LOG_FILE}")

if __name__ == "__main__":
    main()
