#!/usr/bin/env python3
"""
Board Camera - ヘッドレステスト版（スローモーション対応）
ROIフィルタ + 総変化面積ベースの着地検出
"""
import cv2
import numpy as np
import json
from pathlib import Path

VIDEO_FILE  = "test_slow.mp4"
OUTPUT_FILE = "annotated_result.mp4"
LOG_FILE    = Path("test_landings.jsonl")

THROW_THRESH   = 5000   # フレーム間差分がこの値超え → 「投げ中」
QUIET_FRAMES   = 8      # この連続フレーム静止 → 「静止確定」
DIFF_THRESH    = 25     # 静止フレーム間で「新物体」とみなす輝度差
MIN_BLOB_AREA  = 20
MAX_BLOB_AREA  = 800    # ダーツより大きい物体は除外
MIN_TOTAL_AREA = 60     # 総変化面積がこれ未満 → ノイズ
MAX_TOTAL_AREA = 2500   # 総変化面積がこれ超え → 人の動き
WARMUP_FRAMES  = 90
MIN_DETECT_GAP = 60     # 最小検出間隔（フレーム）
BORDER         = 70     # フレーム端N px を除外（ボード外ノイズ）
ROI_BOTTOM_FRAC = 0.5  # フレームの縦この割合より下は無視（ダーツは上半分のみ）


def motion_pixels(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(cv2.absdiff(a, b) > 15))


def find_dart_blob(ref: np.ndarray, current: np.ndarray,
                   existing_pts: list, w: int, h: int):
    """
    ref→current のdiffからダーツ着地点を返す。
    見つからなければ None を返す。
    total_area も返して呼び出し元でスキップ判定に使う。
    """
    diff = cv2.absdiff(current, ref)
    _, bin_ = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    # フレーム端 + 下半分を除外（ダーツは上半分のみ）
    roi_bottom = int(h * ROI_BOTTOM_FRAC)
    border_mask = np.zeros_like(bin_)
    border_mask[BORDER:roi_bottom, BORDER:w-BORDER] = 255
    bin_ = cv2.bitwise_and(bin_, border_mask)

    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if any(abs(cx - px) < 30 and abs(cy - py) < 30 for px, py in existing_pts):
            continue
        blobs.append((area, cx, cy))

    total_area = sum(a for a, _, _ in blobs)
    return blobs, total_area


def main():
    cap   = cv2.VideoCapture(VIDEO_FILE)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 29
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"入力: {w}×{h} @ {fps:.0f}fps  {total}frames ({total/fps:.1f}s)")

    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    prev_gray      = None
    quiet_count    = 0
    in_throw       = False
    ref_frame      = None
    last_detect_fn = -MIN_DETECT_GAP

    last_pts    = []
    log_entries = []
    throw_n     = 0
    frame_n     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        confirmed = None
        status    = "warmup"

        if prev_gray is not None and frame_n > WARMUP_FRAMES:
            mp = motion_pixels(blur, prev_gray)

            if mp > THROW_THRESH:
                quiet_count = 0
                in_throw    = True
                status      = f"THROW mp={mp}"
            else:
                quiet_count += 1
                status      = f"quiet {quiet_count}/{QUIET_FRAMES}"

                if quiet_count == QUIET_FRAMES:
                    if in_throw and ref_frame is not None:
                        since_last = frame_n - last_detect_fn
                        if since_last >= MIN_DETECT_GAP:
                            blobs, total_area = find_dart_blob(
                                ref_frame, blur, last_pts, w, h)
                            tag = f"blobs={len(blobs)} area={total_area:.0f}"
                            t_s = frame_n / fps

                            if MIN_TOTAL_AREA <= total_area <= MAX_TOTAL_AREA and blobs:
                                # 最小面積のブロブ = ダーツ先端候補
                                blobs.sort()
                                _, cx, cy = blobs[0]
                                throw_n += 1
                                confirmed = (cx, cy)
                                last_pts.append(confirmed)
                                last_detect_fn = frame_n
                                entry = {"throw_n": throw_n, "pixel": [cx, cy],
                                         "frame": frame_n, "time_s": round(t_s, 2)}
                                log_entries.append(entry)
                                print(f"[LANDING #{throw_n}] pixel=({cx},{cy})  t={t_s:.1f}s  {tag}")
                            elif total_area > MAX_TOTAL_AREA:
                                print(f"  [SKIP large] {tag}  t={t_s:.1f}s")
                            else:
                                print(f"  [SKIP small] {tag}  t={t_s:.1f}s")

                    ref_frame = blur.copy()
                    in_throw  = False

                elif ref_frame is None and quiet_count >= QUIET_FRAMES:
                    ref_frame = blur.copy()

        prev_gray = blur.copy()

        # --- 描画 ---
        for i, pt in enumerate(last_pts):
            cv2.circle(frame, pt, 9, (0, 0, 255), -1)
            cv2.circle(frame, pt, 9, (255, 255, 255), 1)
            cv2.putText(frame, f"#{i+1}", (pt[0]+8, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        t = frame_n / fps
        cv2.putText(frame, f"t={t:.1f}s  Lands:{len(last_pts)}  {status[:28]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 2)
        prog = frame_n / total
        cv2.rectangle(frame, (0, h-6), (int(w*prog), h), (0, 180, 255), -1)

        out.write(frame)
        if frame_n % 200 == 0:
            print(f"  処理中: {frame_n}/{total} ({100*prog:.0f}%)")

    cap.release()
    out.release()

    with open(LOG_FILE, "w") as f:
        for e in log_entries:
            f.write(json.dumps(e) + "\n")

    print(f"\n=== 結果 ===")
    print(f"処理フレーム: {frame_n}  ({frame_n/fps:.1f}s)")
    print(f"検出着弾数: {throw_n}")
    for e in log_entries:
        print(f"  #{e['throw_n']}  t={e['time_s']}s  pixel={e['pixel']}")
    print(f"出力動画: {OUTPUT_FILE}")
    print(f"ログ: {LOG_FILE}")


if __name__ == "__main__":
    main()
