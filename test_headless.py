#!/usr/bin/env python3
"""
Board Camera - ヘッドレステスト版（ディスプレイ不要）
結果を annotated_result.mp4 として保存
"""
import cv2
import numpy as np
import json
from pathlib import Path

VIDEO_FILE  = "test_clip.mp4"
OUTPUT_FILE = "annotated_result.mp4"
LOG_FILE    = Path("test_landings.jsonl")

class LandingDetector:
    def __init__(self):
        self.bg      = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False)
        self.stable  = 0
        self.pending = None
        self.STABLE  = 8
        self.trail   = []  # 動体軌跡

    def process(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(blur)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = sum(cv2.contourArea(c) for c in contours)

        confirmed = None
        if motion > 150:
            self.stable = 0
            for cnt in contours:
                if cv2.contourArea(cnt) < 50:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                asp = max(w, h) / (min(w, h) + 0.1)
                if asp >= 2.0:
                    pt = (x + w//2, y + h//2)
                    self.pending = pt
                    self.trail.append(pt)
                    if len(self.trail) > 30:
                        self.trail.pop(0)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,80),1)
        else:
            if self.pending:
                self.stable += 1
                if self.stable >= self.STABLE:
                    confirmed    = self.pending
                    self.pending = None
                    self.stable  = 0
                    self.trail   = []

        # 軌跡描画
        for i in range(1, len(self.trail)):
            a = i / len(self.trail)
            cv2.line(frame, self.trail[i-1], self.trail[i],
                     (0, int(255*a), int(255*(1-a))), 1)
        return mask, confirmed

def main():
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 29
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"入力: {w}×{h} @ {fps:.0f}fps  {total}frames ({total/fps:.1f}s)")

    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    detector = LandingDetector()
    throws   = 0
    last_pts = []   # 過去の着弾点すべて
    landings = []

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        mask, confirmed = detector.process(frame)

        if confirmed:
            throws += 1
            last_pts.append(confirmed)
            entry = {"throw_n": throws, "pixel": list(confirmed),
                     "frame": frame_n, "time_s": round(frame_n/fps, 2)}
            landings.append(entry)
            print(f"[LANDING #{throws}] pixel={confirmed}  t={entry['time_s']}s")

        # 全着弾点を描画（累積）
        for i, pt in enumerate(last_pts):
            cv2.circle(frame, pt, 9, (0,0,255), -1)
            cv2.circle(frame, pt, 9, (255,255,255), 1)
            cv2.putText(frame, f"#{i+1}", (pt[0]+8, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,255), 1)

        # HUD
        t = frame_n / fps
        cv2.putText(frame, f"t={t:.1f}s  Landings:{throws}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,0), 2)
        prog = frame_n / total
        cv2.rectangle(frame, (0, h-6), (int(w*prog), h), (0,180,255), -1)

        out.write(frame)
        if frame_n % 100 == 0:
            print(f"  処理中: {frame_n}/{total} ({100*prog:.0f}%)")

    cap.release()
    out.release()

    with open(LOG_FILE, "w") as f:
        for e in landings:
            f.write(json.dumps(e) + "\n")

    print(f"\n=== 結果 ===")
    print(f"処理フレーム: {frame_n}  ({frame_n/fps:.1f}s)")
    print(f"検出着弾数: {throws}")
    for e in landings:
        print(f"  #{e['throw_n']}  t={e['time_s']}s  pixel={e['pixel']}")
    print(f"出力動画: {OUTPUT_FILE}")
    print(f"ログ: {LOG_FILE}")

if __name__ == "__main__":
    main()
