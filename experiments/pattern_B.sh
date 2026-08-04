#!/bin/bash
# pattern_B.sh — 11ドアの2ラウンド目パターン投入
# 内訳: 全閉5 / 全開2 / 半開4
# (door_north_2 と door_south_5 の角度を入れ替え済み)
#
# 使い方:
#   cd ~/lidar_ws/experiments
#   chmod +x pattern_B.sh
#   ./pattern_B.sh
#
# 注意: door_ctrl.py と同じディレクトリ (~/lidar_ws/experiments) から実行すること

set -e

./door_ctrl.py entrance_door 0.0    # 全閉
./door_ctrl.py door_north_1 0.8     # 半開 (expected_free_width≒0.338m)
./door_ctrl.py door_south_1 0.0     # 全閉
./door_ctrl.py door_north_2 0.4     # 半開 (expected_free_width≒0.125m)
./door_ctrl.py door_south_2 0.0     # 全閉
./door_ctrl.py door_north_3 1.3     # 半開 (expected_free_width≒0.746m、境界近く)
./door_ctrl.py door_south_3 1.5708  # 全開
./door_ctrl.py door_north_4 0.0     # 全閉
./door_ctrl.py door_south_4 1.5708  # 全開
./door_ctrl.py door_north_5 0.0     # 全閉
./door_ctrl.py door_south_5 0.7     # 半開 (expected_free_width≒0.273m)

echo "pattern_B 投入完了 (全閉5 / 全開2 / 半開4)"
