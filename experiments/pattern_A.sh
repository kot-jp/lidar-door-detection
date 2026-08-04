#!/bin/bash
# pattern_A.sh — 11ドアの初期パターン投入
# 内訳: 全閉6 / 全開2 / 半開3
#
# 使い方:
#   cd ~/lidar_ws/experiments
#   chmod +x pattern_A.sh
#   ./pattern_A.sh
#
# 注意: door_ctrl.py と同じディレクトリ (~/lidar_ws/experiments) から実行すること

set -e

./door_ctrl.py entrance_door 0.0     # 全閉
./door_ctrl.py door_north_1 0.0      # 全閉
./door_ctrl.py door_south_1 0.0      # 全閉
./door_ctrl.py door_north_2 1.5708   # 全開
./door_ctrl.py door_south_2 0.5      # 半開(浅め)
./door_ctrl.py door_north_3 0.0      # 全閉
./door_ctrl.py door_south_3 1.5708   # 全開
./door_ctrl.py door_north_4 1.3      # 半開(0.7m境界近く)
./door_ctrl.py door_south_4 0.0      # 全閉
./door_ctrl.py door_north_5 0.9      # 半開(中間)
./door_ctrl.py door_south_5 0.0      # 全閉

echo "pattern_A 投入完了 (全閉6 / 全開2 / 半開3)"
