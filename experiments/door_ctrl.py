#!/usr/bin/env python3
"""
door_ctrl.py — ドア開閉コマンド送信 & 正解ラベル記録

使い方:
    ./door_ctrl.py door_north_2 0.8

引数:
    door_name : モデル名 (例: door_north_2, door_south_3, entrance_door)
    angle_rad : ヒンジ角度[rad] (0=全閉 〜 1.5708=全開)

やること:
    1. 指定角度でドアを実際に動かす (ign topic経由)
    2. sim_time・角度・幾何学的に計算した正解通行幅(expected_free_width)を
       /tmp/door_ground_truth.csv に1行追記する

寸法根拠 (corridor10_world.sdf で確認済み, 全11ドア共通):
    FRAME_WIDTH = 1.0m  (壁の開口幅)
    LEAF_LENGTH = 0.95m (扉板の長さ)
    hinge angle=0        -> 扉が壁と平行 (閉)
    hinge angle=1.5708   -> 扉が壁と垂直に開く (全開)
"""

import subprocess
import csv
import sys
import numpy as np
from pathlib import Path
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

GT_LOG_PATH = "/tmp/door_ground_truth.csv"

# corridor10_world.sdf から確認済みの寸法 (全11ドア共通)
FRAME_WIDTH = 1.0    # m
LEAF_LENGTH = 0.95   # m


def expected_free_width(angle_rad: float) -> float:
    """ヒンジ角度から幾何学的な正解通行幅を計算する。
    angle=0 (全閉)     -> frame_width - leaf_length (=0.05m)
    angle=pi/2 (全開)  -> frame_width (=1.0m)
    """
    return max(0.0, FRAME_WIDTH - LEAF_LENGTH * np.cos(angle_rad))


def get_sim_time() -> float:
    """/clock を1回だけsubscribeしてsim_timeを取得する"""
    rclpy.init(args=None)
    node = Node('sim_time_grabber')
    holder = {}

    def cb(msg):
        holder['t'] = msg.clock.sec + msg.clock.nanosec * 1e-9

    node.create_subscription(Clock, '/clock', cb, 10)
    while 't' not in holder:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()
    return holder['t']


def main():
    if len(sys.argv) != 3:
        print(f"使い方: {sys.argv[0]} <door_name> <angle_rad>")
        sys.exit(1)

    door_name = sys.argv[1]
    angle_rad = float(sys.argv[2])

    t = get_sim_time()
    free_width = expected_free_width(angle_rad)

    subprocess.run([
        "ign", "topic", "-t", f"/model/{door_name}/door_cmd",
        "-m", "ignition.msgs.Double", "-p", f"data: {angle_rad}"
    ])

    log_path = Path(GT_LOG_PATH)
    is_new = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['sim_time', 'door_name', 'angle_rad', 'expected_free_width'])
        w.writerow([f'{t:.3f}', door_name, angle_rad, f'{free_width:.3f}'])

    print(f"[{t:.3f}] {door_name} -> angle={angle_rad:.3f}rad "
          f"expected_free_width={free_width:.3f}m")


if __name__ == '__main__':
    main()