# 🎯 Darts Player Tracker

Python + OpenCV によるダーツ投擲追跡アプリ。

プレイヤー側カメラで投げたダーツの**軌跡・リリースポイント・速度**を自動記録します。
ボード認識・スコア計算はスコープ外（別途管理）。

## 機能

- 📷 プレイヤー側カメラでダーツを自動検出
- 🟠 リリースポイント（手から離れた位置）を可視化
- 📈 飛行軌跡をフレームごとに記録
- ⚡ 60fps 対応（Core Ultra 5 推奨）
- 💾 `throws.jsonl` に自動保存

## 必要スペック

| 項目 | 最小 | 推奨 |
|------|------|------|
| CPU | Core i3 / Ryzen 3 | **Intel Core Ultra 5** |
| RAM | 4GB | 8GB |
| カメラ | 60fps 対応 USB カメラ | Logitech C922 / PS Eye |

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python3 player_tracker.py
```

| キー | 操作 |
|------|------|
| `r` | リセット |
| `q` | 終了 |

## 保存データ形式

`throws.jsonl`（1投 = 1行）:

```json
{
  "timestamp": "2026-04-25T21:00:00",
  "release_point": [640, 400],
  "landing_point": [1100, 380],
  "speed_px_s": 1234.5,
  "frame_count": 18,
  "trajectory": [[640,400], [700,398], "..."]
}
```

## 今後の拡張予定

- [ ] YOLOv8 + OpenVINO（Core Ultra 5 NPU 活用）
- [ ] リリース角度・回転数推定
- [ ] マルチカメラ対応（3D軌跡）
