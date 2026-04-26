#!/usr/bin/env python3
"""
Board Camera - ヘッドレステスト版（スローモーション対応）
持続チェック付き着地検出：
  ダーツはボードに刺さったまま残る → N フレーム後も同位置に存在
  手首は通過するだけ          → N フレーム後には消えている
"""
import cv2
import numpy as np
import json
from pathlib import Path

VIDEO_FILE   = "test_slow.mp4"
OUTPUT_FILE  = "annotated_result.mp4"
LOG_FILE     = Path("test_landings.jsonl")

THROW_THRESH     = 5000   # フレーム間差分がこの値超え → 「投げ中」
QUIET_FRAMES     = 8      # 連続静止フレーム数 → 「静止確定」
DIFF_THRESH      = 28     # 着地検出時の輝度差閾値
MIN_BLOB_AREA    = 15
MAX_BLOB_AREA    = 600
MIN_TOTAL_AREA   = 50
MAX_TOTAL_AREA   = 2000
BORDER           = 70     # フレーム端除外（px）
ROI_BOTTOM_FRAC  = 0.5   # 上半分のみ
WARMUP_FRAMES    = 60
MIN_DETECT_GAP   = 50     # 最小検出間隔（フレーム）

# 持続チェック
PERSIST_FRAMES   = 20     # 候補検出からN フレーム後に確認
PERSIST_RADIUS   = 20     # 同位置とみなす半径（px）
PERSIST_THRESH   = 18     # 持続確認時の輝度差閾値


def motion_pixels(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(cv2.absdiff(a, b) > 15))


def find_candidate_blobs(ref: np.ndarray, current: np.ndarray,
                         existing_pts: list, w: int, h: int) -> tuple:
    """静止フレーム差分から候補ブロブを返す"""
    diff = cv2.absdiff(current, ref)
    _, bin_ = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    roi_bottom = int(h * ROI_BOTTOM_FRAC)
    mask = np.zeros_like(bin_)
    mask[BORDER:roi_bottom, BORDER:w - BORDER] = 255
    bin_ = cv2.bitwise_and(bin_, mask)

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


def check_persistence(settle_frame: np.ndarray, check_frame: np.ndarray,
                      cx: int, cy: int) -> bool:
    """(cx,cy) 周辺に PERSIST_FRAMES フレーム後も変化が残っているか確認"""
    r = PERSIST_RADIUS
    y1, y2 = max(0, cy - r), min(settle_frame.shape[0], cy + r)
    x1, x2 = max(0, cx - r), min(settle_frame.shape[1], cx + r)

    patch_settle = settle_frame[y1:y2, x1:x2]
    patch_check  = check_frame[y1:y2, x1:x2]

    diff = cv2.absdiff(patch_check, patch_settle)
    changed = int(np.sum(diff > PERSIST_THRESH))
    return changed > 10  # 10px 以上の変化が残っていれば持続とみなす


def main():
    # 全フレームを先読みして持続チェックに使う
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 29
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"入力: {w}×{h} @ {fps:.0f}fps  {total}frames ({total/fps:.1f}s)")
    print("フレームを先読み中...")

    all_blurs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_blurs.append(cv2.GaussianBlur(gray, (5, 5), 0))
    cap.release()
    print(f"先読み完了: {len(all_blurs)} frames")

    # 通常フレームも先読み（描画用）
    cap2 = cv2.VideoCapture(VIDEO_FILE)
    all_frames = []
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        all_frames.append(frame)
    cap2.release()

    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    prev_gray      = None
    quiet_count    = 0
    in_throw       = False
    ref_frame      = None
    last_detect_fn = -MIN_DETECT_GAP

    # 持続チェック待ちキュー: (check_at_frame, settle_blur, cx, cy, throw_n_tentative)
    pending        = []
    last_pts       = []
    log_entries    = []
    throw_n        = 0

    for frame_n, (blur, frame) in enumerate(zip(all_blurs, all_frames), start=1):
        # 持続チェック
        still_pending = []
        for (check_at, settle_blur, cx, cy) in pending:
            if frame_n >= check_at:
                if check_persistence(settle_blur, blur, cx, cy):
                    throw_n += 1
                    last_pts.append((cx, cy))
                    last_detect_fn = frame_n
                    entry = {"throw_n": throw_n, "pixel": [cx, cy],
                             "frame": frame_n, "time_s": round(frame_n / fps, 2)}
                    log_entries.append(entry)
                    print(f"[LANDING #{throw_n}] pixel=({cx},{cy})  confirmed t={entry['time_s']}s")
                else:
                    print(f"  [REJECT persist] pixel=({cx},{cy})  t={frame_n/fps:.1f}s (消えた=手首)")
            else:
                still_pending.append((check_at, settle_blur, cx, cy))
        pending = still_pending

        status = "warmup"
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
                            blobs, total_area = find_candidate_blobs(
                                ref_frame, blur, last_pts, w, h)
                            t_s = frame_n / fps
                            tag = f"blobs={len(blobs)} area={total_area:.0f}"

                            if MIN_TOTAL_AREA <= total_area <= MAX_TOTAL_AREA and blobs:
                                blobs.sort()
                                _, cx, cy = blobs[0]
                                check_at = min(frame_n + PERSIST_FRAMES, total - 1)
                                pending.append((check_at, blur.copy(), cx, cy))
                                print(f"  [CANDIDATE] pixel=({cx},{cy})  t={t_s:.1f}s  {tag}")
                            elif total_area > MAX_TOTAL_AREA:
                                print(f"  [SKIP large] {tag}  t={t_s:.1f}s")

                    ref_frame = blur.copy()
                    in_throw  = False

                elif ref_frame is None and quiet_count >= QUIET_FRAMES:
                    ref_frame = blur.copy()

        prev_gray = blur.copy()

        # --- 描画 ---
        for i, pt in enumerate(last_pts):
            cv2.circle(frame, pt, 9, (0, 0, 255), -1)
            cv2.circle(frame, pt, 9, (255, 255, 255), 1)
            cv2.putText(frame, f"#{i+1}", (pt[0] + 8, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        t = frame_n / fps
        cv2.putText(frame, f"t={t:.1f}s  Lands:{len(last_pts)}  {status[:28]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 2)
        prog = frame_n / total
        cv2.rectangle(frame, (0, h - 6), (int(w * prog), h), (0, 180, 255), -1)
        out.write(frame)

    out.release()

    with open(LOG_FILE, "w") as f:
        for e in log_entries:
            f.write(json.dumps(e) + "\n")

    print(f"\n=== 結果 ===")
    print(f"処理フレーム: {len(all_blurs)}  ({len(all_blurs)/fps:.1f}s)")
    print(f"検出着弾数: {throw_n}")
    for e in log_entries:
        print(f"  #{e['throw_n']}  t={e['time_s']}s  pixel={e['pixel']}")
    print(f"出力動画: {OUTPUT_FILE}")
    print(f"ログ: {LOG_FILE}")


if __name__ == "__main__":
    main()
