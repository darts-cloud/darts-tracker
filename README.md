# 🎯 Darts Tracker

Python + OpenCV によるダーツ自動追跡アプリ。

カメラで投げたダーツを検出し、座標・スコア・軌跡・リリースポイントを記録します。

## 機能

- 📷 カメラによるダーツ自動検出（背景差分 MOG2）
- 🎯 ダーツボード座標マッピング・スコア自動計算
- 📈 軌跡・リリースポイントの可視化
- 💾 投擲データを `throws.jsonl` に保存
- ⚡ 60fps 対応（Intel Core Ultra / Arc GPU 推奨）

## 必要スペック

| 項目 | 最小 | 推奨 |
|------|------|------|
| CPU | Core i3 / Ryzen 3 | **Core Ultra 5**（NPU対応） |
| RAM | 4GB | 8GB |
| カメラ | 60fps 対応 USB カメラ | Logitech C922 / PS Eye |
| OS | Windows / macOS / Linux | Linux |

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python3 darts_app.py
```

| キー | 操作 |
|------|------|
| `c` | ダーツボード自動検出 |
| `r` | リセット |
| `q` | 終了 |

## データ形式

`throws.jsonl`（1投 = 1行）:

```json
{
  "timestamp": "2026-04-25T21:00:00",
  "board_x": 0.12,
  "board_y": -0.34,
  "score": 60,
  "zone": "treble",
  "release_point": [640, 400],
  "trajectory_points": [[x1,y1], [x2,y2], "..."]
}
```

## 今後の拡張予定

- [ ] YOLOv8 + OpenVINO（NPU推論）による高精度検出
- [ ] ゲームモード（501 / Cricket）
- [ ] 統計・グラフ表示
- [ ] マルチカメラ対応（リリースポイント専用）
