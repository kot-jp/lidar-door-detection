#!/usr/bin/env python3
"""
door_detector_node.py (v5.0)

設計方針:
1. RANSACで壁(直線)を複数抽出する
2. 同一直線上で壁が途切れている箇所(ドア枠候補)を検出する
3. 枠の奥まで見通せるか(beyond)で open/closed の票を集める。
   beyond は True / False / None(判断不能) の3値で、None は投票しない。
4. 枠周辺から壁と非平行な線分(leaf)を探し、直進時の通行幅(horizontal_gap)を出す
5. 複数フレーム観測されたものだけ正式なドアにする(map座標系でトラッキング)

v5.0 の変更点 (v4.x からの差分):
 [A] 考察(論文)の誤検出対策
   A1. 全開ドア→通行幅≒0: leaf無しでも beyond=True なら horizontal_gap=枠幅。
       急角度棄却の近距離しきい値を 1.6rad に(=近距離では棄却しない)。
   A2. 浅い半開の過大評価: leaf探索の垂直距離しきい値を緩め、
       leaf長は「蝶番から最遠点までの距離」で測る。
   A3. open→CLOSED誤検出/チャタリング: beyond を3値化。判断不能は票に入れない。
   A4. 奥のゴーストドア: open候補の見込み角の内側で、より奥にある候補を除外。
   A5. 斜めマーカー: yaw を π 周期で扱う(PCA方向を正規化、平均は2θで取る)。
       RANSACの内点は壁方向に連続した塊だけ採用(遠く離れた同一直線上の点を混ぜない)。
 [B] バグ修正
   B1. _publish_markers の二重呼び出し / orientation.w 未設定
   B2. _merge_walls: recess判定を端点統合より先に行う(recessが壁に飲まれる問題)。
       到達しないコード(d_endpoint<0.3)も削除。
   B3. tf失敗時はそのフレームをスキップ(全トラックが mark_missed されるのを防ぐ)
   B4. _publish_info: 確定ドアが無い時も空文字を流す
 [C] 性能・整理
   C1. PointCloud2 の解析を numpy 化 / RANSAC をベクトル化
   C2. tf は1フレーム1回、点群のstampで引く。TransformListener は別スレッド
   C3. 通行幅の確定は「1m以内の複数フレームの中央値」
   C4. トラックの割当を距離の近い順(貪欲)にし、位置をEMAで平滑化
   C5. ノブ検出・未使用パラメータ・未使用フィールドを削除。パラメータは辞書で一元管理
   C6. 詳細ログは debug レベルに(--ros-args --log-level door_detector:=debug で表示)
"""

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# =========================================================================
# パラメータ(ここだけ見れば全部分かる)
# =========================================================================
PARAMS = {
    # --- 検出範囲・RANSAC ---
    'max_detection_range': 4.0,
    'num_walls': 8,
    'ransac_iterations': 50,
    'ransac_inlier_thresh': 0.05,
    'ransac_seed': 0,                  # 負の値で毎回ランダム
    'min_wall_points': 25,
    'min_wall_length': 0.3,

    # --- ドア寸法・枠 ---
    'door_width_min': 0.5,
    'door_width_max': 1.4,
    'frame_line_dist_thresh': 0.09,
    'frame_dir_sim_thresh': 0.90,

    # --- 壁の統合・recess ---
    'merge_dir_sim_thresh': 0.97,      # 平行とみなす内積(0.97 ≒ 14度)
    'merge_line_dist': 0.08,           # 同一壁とみなす直線間距離
    'wall_endpoint_merge_dist': 0.5,   # 同一壁の分裂とみなす端点距離
    'recess_min_depth': 0.10,
    'recess_max_depth': 0.20,
    'recess_min_length': 0.3,
    'recess_max_length': 1.3,
    'recess_split_min_density': 15.0,  # 点/m

    # --- leaf(扉板)検出 ---
    'leaf_angle_thresh': 0.15,         # これ未満は壁と同じ向きとみなす[rad]
    'leaf_open_angle': 0.30,           # beyond不明でも、この角度以上のleafがあれば open 扱い
    'leaf_perp_ratio': 0.15,           # 蝶番からの距離に掛ける係数(旧0.35)
    'leaf_perp_min': 0.06,             # 垂直距離しきい値の下限(RANSAC許容幅より少し大きく)
    'leaf_reject_angle_near': 1.60,    # 近距離(<=1.5m)の急角度棄却しきい値[rad] (>π/2 なら棄却しない)
    'leaf_reject_angle_far': 1.00,     # 遠距離(>=4m)

    # --- 奥行き(beyond)・通行可否 ---
    'beyond_min_points': 20,
    'beyond_margin': 0.5,
    'passability_confirm_dist': 1.0,
    'passability_min_samples': 3,      # 中央値を取るのに必要な近距離フレーム数
    'passable_width_thresh': 0.7,      # 表示の色分け用

    # --- ゴースト除去 ---
    'ghost_margin': 0.3,

    # --- トラッキング ---
    'track_match_dist': 0.5,
    'track_max_miss': 20,
    'track_confirm_hits': 3,
    'track_width_std_max': 0.20,
    'track_pos_alpha': 0.3,            # 位置のEMA係数(大きいほど最新値を信用)

    # --- 自己除去・高さ ---
    'self_filter_x': 0.50,
    'self_filter_y': 0.45,
    'self_filter_z_min': -0.15,
    'self_filter_z_max': 0.15,
    'wall_height_min': 0.05,
    'wall_height_max': 0.5,
    'floor_z_offset': -0.205,          # lidar_linkから見た床のz

    # --- tf・ログ ---
    'map_frame': 'map',
    'sensor_frame': 'lidar_link',
    'log_csv_path': '/tmp/door_experiment_log.csv',
}


# =========================================================================
# 角度・幾何のヘルパー
# =========================================================================
def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def axial_yaw(yaw: float) -> float:
    """向きの無い直線の角度として [-π/2, π/2) に正規化する(π周期)。"""
    return float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)


def axial_mean(yaws) -> float:
    """π周期の角度の平均(2θで平均してから半分に戻す)。0とπを平均してπ/2になる事故を防ぐ。"""
    y = np.asarray(yaws, dtype=float)
    return float(0.5 * np.arctan2(np.sum(np.sin(2 * y)), np.sum(np.cos(2 * y))))


def blend_axial_yaw(old: float, new: float, alpha: float) -> float:
    s = (1 - alpha) * np.sin(2 * old) + alpha * np.sin(2 * new)
    c = (1 - alpha) * np.cos(2 * old) + alpha * np.cos(2 * new)
    return float(0.5 * np.arctan2(s, c))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


# =========================================================================
# データ構造
# =========================================================================
@dataclass
class Wall:
    points: np.ndarray      # (N,2)
    start: np.ndarray       # 端点1 (XY)
    end: np.ndarray         # 端点2 (XY)
    direction: np.ndarray   # 単位方向ベクトル(x>0 に正規化済み)
    length: float


@dataclass
class DoorCandidate:
    position: np.ndarray                 # 生成時は sensor系、_to_map 後は map系
    width: float
    yaw: float                           # π周期
    source: str = "gap"                  # "gap" / "recess"
    beyond_open: Optional[bool] = None   # True/False/None(判断不能)
    angle_diff: float = -1.0             # leaf検出時のみ >= 0
    horizontal_gap: Optional[float] = None
    view_half_angle: float = 0.0         # 枠を見込む半角(ゴースト除去用)
    leaf_yaw: Optional[float] = None
    leaf_centroid: Optional[np.ndarray] = None
    leaf_length: Optional[float] = None


@dataclass
class DoorTrack:
    position: np.ndarray    # map座標系
    width: float
    yaw: float
    source: str = "gap"
    confirmed: bool = False
    hit_count: int = 0
    miss_count: int = 0
    width_history: deque = field(default_factory=lambda: deque(maxlen=8))
    beyond_history: deque = field(default_factory=lambda: deque(maxlen=8))
    angle_history: deque = field(default_factory=lambda: deque(maxlen=8))
    near_gap_samples: deque = field(default_factory=lambda: deque(maxlen=8))
    last_gap: Optional[float] = None
    state: str = 'unknown'
    passable_confirmed: bool = False
    confirmed_free_width: float = -1.0

    @classmethod
    def from_candidate(cls, c: DoorCandidate) -> "DoorTrack":
        t = cls(position=c.position.copy(), width=c.width, yaw=c.yaw)
        t._push(c)
        t.hit_count = 1
        return t

    def _push(self, c: DoorCandidate):
        self.width = c.width
        self.source = c.source
        self.width_history.append(c.width)
        if c.beyond_open is not None:          # 判断不能(None)は票にしない
            self.beyond_history.append(bool(c.beyond_open))
        if c.angle_diff >= 0:
            self.angle_history.append(c.angle_diff)
        self.last_gap = c.horizontal_gap

    def update(self, c: DoorCandidate, p):
        a = p.track_pos_alpha
        self.position = (1 - a) * self.position + a * c.position
        self.yaw = blend_axial_yaw(self.yaw, c.yaw, a)
        self._push(c)
        self.hit_count += 1
        self.miss_count = 0
        if self.hit_count >= p.track_confirm_hits:
            if len(self.width_history) >= 3:
                recent = list(self.width_history)[-5:]
                self.confirmed = float(np.std(recent)) < p.track_width_std_max
            else:
                self.confirmed = True

    def mark_missed(self):
        self.miss_count += 1
        self.hit_count = max(0, self.hit_count - 1)
        if self.hit_count == 0:
            self.confirmed = False

    def classify(self) -> str:
        if len(self.beyond_history) < 3:
            return self.state

        votes = list(self.beyond_history)[-7:]
        n_open = sum(votes)
        n_closed = len(votes) - n_open

        # leafが安定して大きい角度で見えている → open の追加証拠
        if len(self.angle_history) >= 3:
            angles = list(self.angle_history)[-5:]
            if float(np.std(angles)) < 0.05 and float(np.mean(angles)) > 0.3:
                n_open += 2

        # 僅差(2票未満)は前回の状態を維持(ヒステリシス)
        if abs(n_open - n_closed) < 2:
            return self.state

        self.state = "open" if n_open > n_closed else "closed"
        if self.state != 'open':
            self.passable_confirmed = False
            self.confirmed_free_width = -1.0
            self.near_gap_samples.clear()
        return self.state

    def update_passability(self, dist: float, gap: Optional[float],
                           confirm_dist: float, min_samples: int):
        """open かつ近距離のときだけ gap をサンプルし、中央値で通行幅を確定する。"""
        if self.state != 'open' or gap is None or dist > confirm_dist:
            return
        self.near_gap_samples.append(gap)
        if len(self.near_gap_samples) >= min_samples:
            self.confirmed_free_width = float(np.median(self.near_gap_samples))
            self.passable_confirmed = True


# =========================================================================
# ノード
# =========================================================================
class DoorDetectorNode(Node):

    def __init__(self):
        super().__init__('door_detector')

        for k, v in PARAMS.items():
            self.declare_parameter(k, v)
        self.p = SimpleNamespace(**{k: self.get_parameter(k).value for k in PARAMS})
        self.log = self.get_logger()

        seed = self.p.ransac_seed
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.tracks: Dict[int, DoorTrack] = {}
        self.next_id = 0

        # spin_thread=True: コールバック中でもtfが別スレッドで更新され続ける
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # センサ系は BestEffort で出ていることが多い。sensor_data QoS ならどちらでも受信できる
        self.sub_points = self.create_subscription(
            PointCloud2, '/points', self.points_callback, qos_profile_sensor_data)
        self.pub_markers = self.create_publisher(MarkerArray, '/door_markers', 10)
        self.pub_info = self.create_publisher(String, '/door_info', 10)

        self._log_file = open(Path(self.p.log_csv_path), 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'stamp', 'robot_x', 'robot_y',
            'door_id', 'door_x', 'door_y', 'dist_to_door',
            'source', 'state', 'confirmed', 'hit_count',
            'horizontal_gap', 'passable_confirmed', 'confirmed_free_width'
        ])

        self.log.info('DoorDetector v5.0 started')

    # =====================================================================
    # メインコールバック
    # =====================================================================
    def points_callback(self, msg: PointCloud2):
        p = self.p

        pose = self._get_sensor_pose(msg.header.stamp)
        if pose is None:
            # tfが無いフレームは丸ごとスキップ(候補0扱いで mark_missed しない)
            self.log.warn('tf取得失敗: このフレームをスキップ', throttle_duration_sec=2.0)
            return
        robot_pos, _ = pose

        points = self._parse_pointcloud2(msg)
        if points is None or len(points) < 50:
            return
        points = self._remove_self(points)
        if len(points) < 50:
            return
        points = points[np.hypot(points[:, 0], points[:, 1]) <= p.max_detection_range]
        if len(points) < 50:
            return

        rel_z = points[:, 2] - p.floor_z_offset          # 床からの高さ
        band = (rel_z >= p.wall_height_min) & (rel_z <= p.wall_height_max)
        band_xy = points[band][:, :2]                    # 壁・扉板用(床から5〜50cm)
        scene_xy = points[rel_z >= p.wall_height_min][:, :2]   # 奥行き判定用(床の点は除く)
        if len(band_xy) < 50:
            return

        walls = self._extract_walls(band_xy)
        walls, recess_walls = self._merge_walls(walls)
        candidates = self._find_door_candidates(walls, scene_xy, band_xy, recess_walls)
        candidates = self._to_map(candidates, pose)
        for c in candidates:
            self.log.debug(
                f"cand: map=({c.position[0]:.2f},{c.position[1]:.2f}) src={c.source} "
                f"beyond={c.beyond_open} leaf={'有' if c.angle_diff >= 0 else '無'} "
                f"gap={c.horizontal_gap} robot_x={robot_pos[0]:.2f}")

        self._update_tracks(candidates)
        self._prune_tracks()

        confirmed = [(tid, t) for tid, t in self.tracks.items() if t.confirmed]

        for tid, t in confirmed:
            t.classify()
            if t.miss_count == 0 and t.state == 'open':   # 今フレームで観測できたものだけ
                dist_now = float(np.linalg.norm(t.position - robot_pos))
                was = t.passable_confirmed
                t.update_passability(dist_now, t.last_gap,
                                     p.passability_confirm_dist, p.passability_min_samples)
                if t.passable_confirmed and not was:
                    self.log.info(f"Door{tid} passable確定: dist={dist_now:.2f} "
                                  f"gap={t.confirmed_free_width:.2f}")

        self._log_experiment_data(msg.header.stamp, confirmed, robot_pos)
        self._publish_markers(confirmed)
        self._publish_info(confirmed)

        n_closed = sum(1 for _, t in confirmed if t.state == 'closed')
        n_open = sum(1 for _, t in confirmed if t.state == 'open')
        self.log.info(
            f'walls={len(walls)} candidates={len(candidates)} confirmed={len(confirmed)} '
            f'closed={n_closed} open={n_open}', throttle_duration_sec=2.0)

    # =====================================================================
    # tf
    # =====================================================================
    def _get_sensor_pose(self, stamp) -> Optional[Tuple[np.ndarray, float]]:
        """map系でのセンサの (xy, yaw)。点群のstampで引き、無理なら最新で代用。"""
        p = self.p
        tf = None
        for t in (Time.from_msg(stamp), Time()):
            try:
                tf = self.tf_buffer.lookup_transform(p.map_frame, p.sensor_frame, t)
                break
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                continue
        if tf is None:
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        yaw = float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        return np.array([tr.x, tr.y]), yaw

    def _to_map(self, candidates: List[DoorCandidate], pose) -> List[DoorCandidate]:
        """候補の位置とyawを sensor系 → map系 に変換(roll/pitch無しの2D前提)。"""
        t_xy, yaw0 = pose
        c, s = np.cos(yaw0), np.sin(yaw0)
        R = np.array([[c, -s], [s, c]])
        for cand in candidates:
            cand.position = R @ cand.position + t_xy
            cand.yaw = axial_yaw(cand.yaw + yaw0)
        return candidates

    # =====================================================================
    # 前処理
    # =====================================================================
    def _parse_pointcloud2(self, msg: PointCloud2) -> Optional[np.ndarray]:
        f = {x.name: x for x in msg.fields}
        if not all(k in f for k in ('x', 'y', 'z')):
            return None
        if any(f[k].datatype != 7 for k in ('x', 'y', 'z')):   # 7 = FLOAT32
            self.log.warn('x/y/z が FLOAT32 ではありません', throttle_duration_sec=5.0)
            return None
        n = msg.width * msg.height
        if n == 0:
            return None

        e = '>' if msg.is_bigendian else '<'
        dt = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [e + 'f4'] * 3,
            'offsets': [f['x'].offset, f['y'].offset, f['z'].offset],
            'itemsize': msg.point_step,
        })
        arr = np.frombuffer(msg.data, dtype=dt, count=n)
        # 将来intensityを使うときは、ここに 'intensity' を足して (N,4) にする
        pts = np.column_stack((arr['x'], arr['y'], arr['z'])).astype(np.float32)
        return pts[np.isfinite(pts).all(axis=1)]

    def _remove_self(self, points: np.ndarray) -> np.ndarray:
        p = self.p
        inside = (
            (np.abs(points[:, 0]) < p.self_filter_x) &
            (np.abs(points[:, 1]) < p.self_filter_y) &
            (points[:, 2] > p.self_filter_z_min) &
            (points[:, 2] < p.self_filter_z_max)
        )
        return points[~inside]

    # =====================================================================
    # 壁抽出 (RANSACをベクトル化)
    # =====================================================================
    def _extract_walls(self, xy: np.ndarray) -> List[Wall]:
        p = self.p
        walls: List[Wall] = []
        remaining = xy.copy()
        # 内点がこれ以上離れていたら別の壁とみなす(ドア枠のgapは許容したいので door_width_max より少し大きく)
        split_gap = p.door_width_max + 0.2

        for _ in range(p.num_walls):
            n = len(remaining)
            if n < p.min_wall_points:
                break

            # 仮説をまとめて作る
            idx = self.rng.integers(0, n, size=(p.ransac_iterations, 2))
            idx = idx[idx[:, 0] != idx[:, 1]]
            if len(idx) == 0:
                break
            p1 = remaining[idx[:, 0]]
            d = remaining[idx[:, 1]] - p1
            norm = np.linalg.norm(d, axis=1)
            ok = norm > 1e-6
            p1, d, norm = p1[ok], d[ok], norm[ok]
            if len(p1) == 0:
                break
            d = d / norm[:, None]

            diff = remaining[None, :, :] - p1[:, None, :]                       # (K,N,2)
            perp = np.abs(diff[..., 0] * d[:, None, 1] - diff[..., 1] * d[:, None, 0])   # (K,N)
            masks = perp < p.ransac_inlier_thresh
            counts = masks.sum(axis=1)
            best = int(np.argmax(counts))
            if counts[best] < p.min_wall_points:
                break

            # 内点を壁方向に並べ、大きな空白で分けて最大の塊だけを壁にする
            inl = np.flatnonzero(masks[best])
            proj = (remaining[inl] - p1[best]) @ d[best]
            order = np.argsort(proj)
            inl = inl[order]
            cuts = np.flatnonzero(np.diff(proj[order]) > split_gap) + 1
            seg = max(np.split(inl, cuts), key=len)

            if len(seg) >= p.min_wall_points:
                wall = self._fit_wall(remaining[seg])
                if wall is not None and wall.length >= p.min_wall_length:
                    walls.append(wall)
                    self.log.debug(f"wall: n={len(seg)} len={wall.length:.2f} "
                                   f"({wall.start[0]:.2f},{wall.start[1]:.2f})->"
                                   f"({wall.end[0]:.2f},{wall.end[1]:.2f})")

            keep = np.ones(n, dtype=bool)
            keep[seg] = False
            remaining = remaining[keep]

        return walls

    def _fit_wall(self, pts: np.ndarray) -> Optional[Wall]:
        if len(pts) < 2:
            return None
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        if cov.ndim < 2:
            return None
        _, vecs = np.linalg.eigh(cov)
        direction = vecs[:, -1]                      # 最大固有値の方向
        # 固有ベクトルの符号は不定 → x>0 側に揃える(frameごとに反転しないように)
        if direction[0] < -1e-9 or (abs(direction[0]) <= 1e-9 and direction[1] < 0):
            direction = -direction
        proj = (pts - center) @ direction
        return Wall(
            points=pts,
            start=center + proj.min() * direction,
            end=center + proj.max() * direction,
            direction=direction,
            length=float(proj.max() - proj.min()),
        )

    def _line_dist(self, base: Wall, w: Wall) -> float:
        """w の両端点から base の直線までの垂直距離の平均。"""
        d1 = abs(cross2(base.direction, w.start - base.start))
        d2 = abs(cross2(base.direction, w.end - base.start))
        return 0.5 * (d1 + d2)

    # =====================================================================
    # 壁の統合 / recess 分離
    # =====================================================================
    def _merge_walls(self, walls: List[Wall]):
        p = self.p
        if len(walls) <= 1:
            return walls, []

        merged: List[Wall] = []
        recess: List[Wall] = []
        used = [False] * len(walls)

        for i, base in enumerate(walls):
            if used[i]:
                continue
            used[i] = True
            cluster = [base]

            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                w = walls[j]
                if abs(base.direction @ w.direction) < p.merge_dir_sim_thresh:
                    continue

                line_dist = self._line_dist(base, w)

                # ① 先に recess(奥まった平行壁=閉じたドア板候補)を判定
                if p.recess_min_depth <= line_dist <= p.recess_max_depth:
                    recess.extend(self._split_recess_wall(w))
                    used[j] = True
                    continue

                # ② ほぼ同一直線で、端点が近ければ「同じ壁の分裂」として統合
                if line_dist > p.merge_line_dist:
                    continue
                d_end = min(
                    np.linalg.norm(base.start - w.start), np.linalg.norm(base.start - w.end),
                    np.linalg.norm(base.end - w.start), np.linalg.norm(base.end - w.end),
                )
                if d_end < p.wall_endpoint_merge_dist:
                    cluster.append(w)
                    used[j] = True

            if len(cluster) == 1:
                merged.append(base)
            else:
                fit = self._fit_wall(np.vstack([w.points for w in cluster]))
                merged.append(fit if fit is not None else base)

        return merged, recess

    def _split_recess_wall(self, w: Wall) -> List[Wall]:
        """奥まった長い1本を、点の空白(door_width_max超)でドア1枚ぶんずつに分ける。"""
        p = self.p
        proj = (w.points - w.start) @ w.direction
        order = np.argsort(proj)
        pts = w.points[order]
        cuts = np.flatnonzero(np.diff(proj[order]) > p.door_width_max) + 1

        result = []
        for seg in np.split(pts, cuts):
            if len(seg) < 8:
                continue
            fit = self._fit_wall(seg)
            if fit is None or not (p.recess_min_length <= fit.length <= p.recess_max_length):
                continue
            density = len(seg) / max(fit.length, 0.05)
            if density < p.recess_split_min_density:
                self.log.debug(f"recess棄却(密度): len={fit.length:.2f} n={len(seg)} {density:.1f}pts/m")
                continue
            result.append(fit)
        self.log.debug(f"recess split: 元len={w.length:.2f} -> {len(result)}区画")
        return result

    # =====================================================================
    # ドア候補
    # =====================================================================
    def _find_door_candidates(self, walls, scene_xy, band_xy, recess_walls) -> List[DoorCandidate]:
        frames = self._find_door_frames(walls)
        frame_cands = [self._classify_frame(f, scene_xy, band_xy) for f in frames]

        trusted = [c for c in frame_cands if c.angle_diff >= 0]      # leaf検出あり
        untrusted = [c for c in frame_cands if c.angle_diff < 0]
        dedup_dist = 0.4

        final = list(trusted)

        # recess: leaf由来と重複するもの/信頼できるframe候補に近いものは捨てる
        leaves = [(c.leaf_yaw, c.leaf_centroid, c.leaf_length) for c in trusted]
        recess_walls = [w for w in recess_walls if not self._wall_matches_leaf(w, leaves)]
        recess_cands = []
        for rc in self._recess_to_candidates(recess_walls):
            if any(np.linalg.norm(rc.position - c.position) < dedup_dist for c in trusted):
                continue
            recess_cands.append(rc)
        final += recess_cands

        # leaf無しframe候補: recessに近いものは捨てる。
        # 遠距離(>1.5m)は、奥が見えている(beyond=True)場合以外は信用しない
        for uc in untrusted:
            if any(np.linalg.norm(uc.position - rc.position) < dedup_dist for rc in recess_cands):
                continue
            if uc.beyond_open is not True and np.linalg.norm(uc.position) > 1.5:
                continue
            final.append(uc)

        return self._suppress_ghosts(final)

    def _suppress_ghosts(self, cands: List[DoorCandidate]) -> List[DoorCandidate]:
        """open と判断できる枠の「向こう側」に見えている候補は、奥の部屋の壁の途切れ
        (ゴースト)の可能性が高いので除外する。"""
        margin = self.p.ghost_margin
        occluders = [
            (np.arctan2(c.position[1], c.position[0]), float(np.linalg.norm(c.position)), c.view_half_angle)
            for c in cands if c.beyond_open is True and c.view_half_angle > 0
        ]
        out = []
        for c in cands:
            ang = np.arctan2(c.position[1], c.position[0])
            dist = float(np.linalg.norm(c.position))
            ghost = any(abs(wrap_angle(ang - oa)) < oh and dist > od + margin
                        for oa, od, oh in occluders)
            if ghost:
                self.log.debug(f"ghost除外: ({c.position[0]:.2f},{c.position[1]:.2f}) dist={dist:.2f}")
                continue
            out.append(c)
        return out

    def _wall_matches_leaf(self, wall: Wall, leaves) -> bool:
        """recess壁が、検出済みの開いた扉板(leaf)と同一物かを判定する。"""
        center = (wall.start + wall.end) / 2.0
        for leaf_yaw, leaf_centroid, leaf_length in leaves:
            if leaf_centroid is None or leaf_yaw is None:
                continue
            leaf_dir = np.array([np.cos(leaf_yaw), np.sin(leaf_yaw)])
            if abs(wall.direction @ leaf_dir) < 0.85:
                continue
            dist = float(np.linalg.norm(center - leaf_centroid))
            if dist < max(leaf_length, wall.length) * 0.6 + 0.3:
                self.log.debug(f"recess壁をleaf由来として除外: dist={dist:.2f}")
                return True
        return False

    def _recess_to_candidates(self, recess_walls: List[Wall]) -> List[DoorCandidate]:
        """奥まった壁(=閉じたドア板)は closed の証拠として beyond_open=False で登録する。"""
        out = []
        for w in recess_walls:
            out.append(DoorCandidate(
                position=(w.start + w.end) / 2.0, width=w.length,
                yaw=float(np.arctan2(w.direction[1], w.direction[0])),
                source="recess", beyond_open=False,
            ))
        return out

    # =====================================================================
    # ドア枠(同一直線上の壁の分断)
    # =====================================================================
    def _find_door_frames(self, walls: List[Wall]) -> List[dict]:
        p = self.p
        frames = []
        for group in self._group_collinear_walls(walls):
            pts = np.vstack([w.points for w in group])
            direction = group[0].direction
            base = group[0].start

            proj = np.sort((pts - base) @ direction)
            if len(proj) < 2:
                continue
            gaps = np.diff(proj)
            for i in np.flatnonzero((gaps >= p.door_width_min) & (gaps <= p.door_width_max)):
                if not self._has_solid_support(proj, int(i)):
                    continue
                a = base + proj[i] * direction
                b = base + proj[i + 1] * direction
                frames.append(dict(
                    edge_a=a, edge_b=b, frame_width=float(gaps[i]),
                    wall_yaw=float(np.arctan2(direction[1], direction[0])),
                    center=(a + b) / 2.0,
                ))
        return self._merge_frames(frames)

    @staticmethod
    def _has_solid_support(proj_sorted: np.ndarray, gap_idx: int,
                           window: float = 0.15, min_support: int = 4) -> bool:
        """gapの両端それぞれ window 以内に min_support 点以上あるか(孤立点が作ったgapを除く)。"""
        left = proj_sorted[gap_idx]
        right = proj_sorted[gap_idx + 1]
        n_left = np.sum((proj_sorted >= left - window) & (proj_sorted <= left))
        n_right = np.sum((proj_sorted >= right) & (proj_sorted <= right + window))
        return n_left >= min_support and n_right >= min_support

    def _group_collinear_walls(self, walls: List[Wall]) -> List[List[Wall]]:
        p = self.p
        groups, used = [], [False] * len(walls)
        for i, wa in enumerate(walls):
            if used[i]:
                continue
            used[i] = True
            group = [wa]
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                wb = walls[j]
                if abs(wa.direction @ wb.direction) < p.frame_dir_sim_thresh:
                    continue
                if self._line_dist(wa, wb) > p.frame_line_dist_thresh:
                    continue
                group.append(wb)
                used[j] = True
            groups.append(group)
        return groups

    def _merge_frames(self, frames: List[dict]) -> List[dict]:
        merged, used = [], [False] * len(frames)
        for i, f in enumerate(frames):
            if used[i]:
                continue
            used[i] = True
            cluster = [f]
            for j in range(i + 1, len(frames)):
                if not used[j] and np.linalg.norm(frames[j]['center'] - f['center']) < 0.3:
                    used[j] = True
                    cluster.append(frames[j])
            m = dict(f)   # edge / center / width は先頭のものに統一(互いにズレないように)
            m['wall_yaw'] = axial_mean([c['wall_yaw'] for c in cluster])
            merged.append(m)
        return merged

    # =====================================================================
    # 枠の分類: beyond(奥行き) + leaf(扉板)
    # =====================================================================
    def _classify_frame(self, frame: dict, scene_xy: np.ndarray, band_xy: np.ndarray) -> DoorCandidate:
        dist = float(np.linalg.norm(frame['center']))
        fw = frame['frame_width']

        beyond, half_angle = self._detect_beyond(frame, scene_xy, dist)

        leaf = self._detect_leaf(frame, band_xy)
        if leaf is not None and leaf['angle_diff'] > self._angle_reject_thresh(dist):
            self.log.debug(f"leaf棄却(急角度): dist={dist:.2f} angle={leaf['angle_diff']:.2f}")
            leaf = None

        common = dict(position=frame['center'].copy(), width=fw, yaw=frame['wall_yaw'],
                      source="gap", view_half_angle=half_angle)

        if leaf is None:
            # leaf無しでも奥が見えていれば「扉板が見えないほど開いている(全開・エッジオン)」とみなし、
            # 通行幅は枠幅とする。奥が不明/closedなら通行幅は不明(None)。0にはしない。
            gap = fw if beyond is True else None
            return DoorCandidate(**common, beyond_open=beyond, horizontal_gap=gap)

        angle = leaf['angle_diff']
        if beyond is not True and self.p.leaf_open_angle <= angle <= 1.4:
            beyond = True      # 奥が見えなくても、はっきり開いた扉板が見えている

        # 直進時の実効通行幅 = 枠幅 − 扉板を壁方向に射影した長さ。
        # beyond=False(閉じている)のときは、leafが拾えても通行幅という概念自体が
        # 意味を持たないため None にする(誤って大きな値が記録・集計されるのを防ぐ)。
        if beyond is False:
            gap = None
        else:
            gap = max(0.0, fw - leaf['leaf_length'] * float(np.cos(angle)))
        self.log.debug(f"frame: dist={dist:.2f} beyond={beyond} angle={angle:.2f} "
                       f"leaf_len={leaf['leaf_length']:.2f} gap={gap}")

        return DoorCandidate(
            **common, beyond_open=beyond, angle_diff=angle, horizontal_gap=gap,
            leaf_yaw=leaf['leaf_yaw'], leaf_centroid=leaf['leaf_centroid'],
            leaf_length=leaf['leaf_length'],
        )

    def _angle_reject_thresh(self, dist: float) -> float:
        p = self.p
        return float(np.interp(dist, [1.5, 4.0],
                               [p.leaf_reject_angle_near, p.leaf_reject_angle_far]))

    def _required_points(self, dist: float) -> int:
        p = self.p
        far = max(8, p.beyond_min_points // 2)
        return int(round(np.interp(dist, [1.5, 4.0], [p.beyond_min_points, far])))

    def _frame_view(self, frame: dict, wall_dist: float) -> Tuple[float, float]:
        """枠の方位角と、枠を見込む半角(斜めから見るほど細く見える補正つき)。"""
        center = frame['center']
        frame_angle = float(np.arctan2(center[1], center[0]))
        wall_normal = frame['wall_yaw'] + np.pi / 2.0
        viewing = wrap_angle(frame_angle - wall_normal)
        foreshorten = max(abs(np.cos(viewing)), 0.15)
        half = float(np.arctan2(frame['frame_width'] * foreshorten * 0.5, max(wall_dist, 0.1)))
        return frame_angle, half

    @staticmethod
    def _depth_clusters(distances: np.ndarray, gap_thresh: float):
        """距離を1次元で並べ、gap_thresh以上の空白で塊に分ける。[(平均, 点数, std), ...] 近い順"""
        if len(distances) == 0:
            return []
        d = np.sort(distances)
        cuts = np.flatnonzero(np.diff(d) > gap_thresh) + 1
        return [(float(s.mean()), len(s), float(s.std())) for s in np.split(d, cuts)]

    def _detect_beyond(self, frame: dict, scene_xy: np.ndarray, wall_dist: float
                       ) -> Tuple[Optional[bool], float]:
        """枠の向こうまで点が届いているか。 True=open / False=closed / None=判断不能"""
        p = self.p
        frame_angle, half = self._frame_view(frame, wall_dist)

        ang = np.arctan2(scene_xy[:, 1], scene_xy[:, 0])
        d_ang = np.abs(np.arctan2(np.sin(ang - frame_angle), np.cos(ang - frame_angle)))
        win = np.hypot(scene_xy[:, 0], scene_xy[:, 1])[d_ang < half]

        required = self._required_points(wall_dist)
        min_obs = max(5, required // 2)
        if len(win) < min_obs:
            return None, half                       # 見えてない=分からない

        clusters = self._depth_clusters(win, p.beyond_margin * 0.6)
        near_limit = wall_dist + p.beyond_margin
        near_cnt = int(np.sum(win <= near_limit))
        beyond = [c for c in clusters if c[0] > near_limit]

        if beyond:
            mean_d, cnt, std = beyond[0]
            if cnt >= required and std < 0.5:
                # 壁と奥の間に別の塊(扉板など)があるなら、ここでは断定しない
                inter = [c for c in clusters
                         if wall_dist + 0.05 < c[0] <= near_limit and c[1] >= 5]
                if inter:
                    self.log.debug(f"beyond保留(中間構造あり): {inter}")
                    return None, half
                self.log.debug(f"beyond確定: mean={mean_d:.2f} n={cnt} std={std:.2f}")
                return True, half

        # 奥まで届いていない。壁の点が十分見えているときだけ「閉じてる」と言う
        if near_cnt >= min_obs:
            return False, half
        return None, half

    # =====================================================================
    # leaf(扉板)検出
    # =====================================================================
    def _detect_leaf(self, frame: dict, band_xy: np.ndarray) -> Optional[dict]:
        p = self.p
        fw = frame['frame_width']
        wall_dir = np.array([np.cos(frame['wall_yaw']), np.sin(frame['wall_yaw'])])

        best = None
        for hinge in (frame['edge_a'], frame['edge_b']):    # それぞれ「ここが蝶番」と仮定
            diff = band_xy - hinge
            d_h = np.linalg.norm(diff, axis=1)
            perp = np.abs(diff[:, 0] * wall_dir[1] - diff[:, 1] * wall_dir[0])

            # 蝶番から遠いほど厳しく。浅い開き角でも拾えるよう、係数と下限を小さめにしてある
            thresh = np.clip(p.leaf_perp_ratio * d_h, p.leaf_perp_min, p.recess_max_depth)
            near = band_xy[(d_h < fw * 1.3) & (perp > thresh)]
            if len(near) < 15:
                continue

            cluster = self._largest_cluster(near, eps=0.08)
            if cluster is None or len(cluster) < 10:
                continue
            fit = self._fit_wall(cluster)
            if fit is None or fit.length > fw * 1.5 or fit.length < 0.15:
                continue

            angle = float(np.arccos(np.clip(abs(fit.direction @ wall_dir), 0.0, 1.0)))
            if angle < p.leaf_angle_thresh:
                continue

            d_cluster = np.linalg.norm(cluster - hinge, axis=1)
            attach = float(d_cluster.min())          # 蝶番に近いほど「そこに付いてる扉板」らしい
            cand = dict(
                leaf_yaw=float(np.arctan2(fit.direction[1], fit.direction[0])),
                leaf_length=float(d_cluster.max()),  # 蝶番から最遠点まで(近い側が欠けても長さを過小評価しない)
                angle_diff=angle,
                leaf_centroid=cluster.mean(axis=0),
                attach=attach,
            )
            # ★追加：候補が見つかるたびに詳細をログに出す
            self.log.debug(
                f"leaf候補: hinge=({hinge[0]:.2f},{hinge[1]:.2f}) n_near={len(near)} "
                f"n_cluster={len(cluster)} fit_len={fit.length:.2f} "
                f"leaf_length={cand['leaf_length']:.2f} attach={attach:.2f} angle={angle:.2f}"
            )
            if best is None or attach < best['attach']:
                best = cand

        if best is None:
            self.log.debug("leaf: 候補なし")
        return best

    def _largest_cluster(self, pts: np.ndarray, eps: float = 0.15) -> Optional[np.ndarray]:
        """距離eps以内を連結した最大の塊(隣接行列を作ってまとめて計算)。"""
        n = len(pts)
        if n == 0:
            return None
        if n > 1500:
            pts = pts[self.rng.choice(n, 1500, replace=False)]
            n = 1500

        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        adj = d2 < eps * eps

        labels = -np.ones(n, dtype=int)
        cur = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = cur
            stack = [i]
            while stack:
                k = stack.pop()
                nb = np.flatnonzero(adj[k] & (labels < 0))
                labels[nb] = cur
                stack.extend(nb.tolist())
            cur += 1

        return pts[labels == int(np.argmax(np.bincount(labels)))]

    # =====================================================================
    # トラッキング
    # =====================================================================
    def _update_tracks(self, candidates: List[DoorCandidate]):
        p = self.p

        # 距離の近い組から順に割り当てる(候補の並び順に依存しない)
        pairs = []
        for ci, c in enumerate(candidates):
            for tid, t in self.tracks.items():
                d = float(np.linalg.norm(c.position - t.position))
                if d < p.track_match_dist:
                    pairs.append((d, ci, tid))
        pairs.sort()

        used_c, used_t = set(), set()
        for _, ci, tid in pairs:
            if ci in used_c or tid in used_t:
                continue
            self.tracks[tid].update(candidates[ci], p)
            used_c.add(ci)
            used_t.add(tid)

        for ci, c in enumerate(candidates):
            if ci in used_c:
                continue
            self.tracks[self.next_id] = DoorTrack.from_candidate(c)
            used_t.add(self.next_id)
            self.next_id += 1

        for tid, t in self.tracks.items():
            if tid not in used_t:
                t.mark_missed()

    def _prune_tracks(self):
        for tid in [tid for tid, t in self.tracks.items() if t.miss_count > self.p.track_max_miss]:
            del self.tracks[tid]

    # =====================================================================
    # 出力
    # =====================================================================
    def _state_color(self, t: DoorTrack) -> Tuple[float, float, float]:
        if t.state == 'open' and t.passable_confirmed:
            if t.confirmed_free_width >= self.p.passable_width_thresh:
                return 0.0, 0.5, 1.0     # 青: 通行確定
            return 1.0, 0.55, 0.0        # オレンジ: 開いてるが幅不足
        return {'open': (0.0, 1.0, 0.0), 'closed': (1.0, 0.0, 0.0)}.get(t.state, (1.0, 1.0, 0.0))

    def _publish_markers(self, confirmed: list):
        frame_id = self.p.map_frame
        now = self.get_clock().now().to_msg()

        ma = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for tid, t in confirmed:
            half_w = t.width / 2.0
            dx, dy = np.cos(t.yaw) * half_w, np.sin(t.yaw) * half_w

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = now
            m.ns = 'door_frame'
            m.id = tid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            m.color.r, m.color.g, m.color.b = self._state_color(t)
            m.color.a = 1.0
            p1, p2 = Point(), Point()
            p1.x, p1.y = float(t.position[0] - dx), float(t.position[1] - dy)
            p2.x, p2.y = float(t.position[0] + dx), float(t.position[1] + dy)
            m.points = [p1, p2]
            ma.markers.append(m)

            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = now
            label.ns = 'door_label'
            label.id = tid
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.orientation.w = 1.0
            label.pose.position.x = float(t.position[0])
            label.pose.position.y = float(t.position[1])
            label.pose.position.z = 0.5
            label.scale.z = 0.2
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            if t.passable_confirmed:
                label.text = f"Door{tid}: {t.state.upper()}\ngap={t.confirmed_free_width:.2f}m"
            else:
                label.text = f"Door{tid}: {t.state.upper()}\nw={t.width:.2f}m"
            ma.markers.append(label)

        self.pub_markers.publish(ma)

    def _publish_info(self, confirmed: list):
        msg = String()
        # 確定ドアが無いときも空文字を流す(購読側が古い情報を握り続けないように)
        msg.data = '\n'.join(
            f"Door{tid}: {t.state} pos=({t.position[0]:.2f},{t.position[1]:.2f}) width={t.width:.2f}m"
            for tid, t in confirmed
        )
        self.pub_info.publish(msg)

    def _log_experiment_data(self, stamp, confirmed, robot_pos):
        ts = stamp.sec + stamp.nanosec * 1e-9
        for tid, t in confirmed:
            dist = float(np.linalg.norm(t.position - robot_pos))
            gap = t.last_gap if t.last_gap is not None else -1.0
            self._log_writer.writerow([
                f'{ts:.3f}', f'{robot_pos[0]:.3f}', f'{robot_pos[1]:.3f}',
                tid, f'{t.position[0]:.3f}', f'{t.position[1]:.3f}', f'{dist:.3f}',
                t.source, t.state, t.confirmed, t.hit_count,
                f'{gap:.3f}', t.passable_confirmed, f'{t.confirmed_free_width:.3f}'
            ])
        self._log_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._log_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()