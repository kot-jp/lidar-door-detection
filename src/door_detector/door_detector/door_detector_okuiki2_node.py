#!/usr/bin/env python3
"""
door_detector_node.py (v4.1)

設計方針(v4):
1. RANSACで壁(直線)を複数抽出する
2. 同一直線上にある壁が途切れている箇所(ドア枠候補)を検出する
   (v3の「gapがあるか」ではなく「壁が分断されているか」で判定。
    開き戸は閉じてると壁と面一になりgapが出ないため)
3. 枠候補の周辺点群から、壁と非平行な線分(リーフ)を検出する。
   リーフが見つからなければ closed、見つかればその角度から
   蝶番位置・リーフ長を使って幾何学的に「実際に通れる幅(free_width)」
   を計算する。
4. free_width をロボット幅(60cm)を基準にした閾値(0.7m)で
   open / closed に分類する。
5. トラッキング: 新規候補は仮登録とし、複数フレーム連続で
   観測されたものだけ正式なドアとして扱う。
   一定フレーム観測が無ければ自動的に削除する。

v4.1での変更:
- tf2導入。ドア候補位置を lidar_link -> map に変換してからトラッキングする。
  これによりロボットが移動してもドアの同一性を正しく維持できる。
- マーカーのframe_idもmapに変更。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
import numpy as np
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
import rclpy.time
import tf2_ros
from tf2_geometry_msgs import do_transform_point


@dataclass
class Wall:
    points: np.ndarray      # この壁を構成する点群(N,3)
    start: np.ndarray       # 端点1 (XY)
    end: np.ndarray         # 端点2 (XY)
    direction: np.ndarray   # 単位方向ベクトル(XY)
    length: float


@dataclass
class DoorCandidate:
    position: np.ndarray    # XY (map座標系)
    width: float            # 枠幅(frame_width)
    yaw: float
    gap_score: float        # free_width と同義(通行可能幅)
    depth_score: float      # free_width と同義(通行可能幅、historyに積む用)
    height_score: float     # v4では未使用、1.0固定(将来の拡張用に残す)
    occupancy_score: float  # v4では未使用、0/1固定(将来の拡張用に残す)
    angle_diff: float = -1.0  # v4.6: リーフ検出時の角度差(未検出なら-1)
    total_score: float = 0.0
    beyond_open: bool = False   # ★追加：奥行き判定の結果
    horizontal_gap: float = 0.0   # ★追加：直進時に実際に使える通行幅

@dataclass
class DoorTrack:
    position: np.ndarray    # map座標系
    width: float
    yaw: float
    confirmed: bool = False
    hit_count: int = 0
    miss_count: int = 0
    width_history: deque = field(default_factory=lambda: deque(maxlen=8))
    depth_history: deque = field(default_factory=lambda: deque(maxlen=8))  # free_width の履歴
    occupancy_history: deque = field(default_factory=lambda: deque(maxlen=8))
    angle_history: deque = field(default_factory=lambda: deque(maxlen=8))  # v4.6: angle_diffの履歴
    beyond_history: deque = field(default_factory=lambda: deque(maxlen=8))  
    horizontal_gap_history: deque = field(default_factory=lambda: deque(maxlen=8)) 
    state: str = 'unknown'
    passable_confirmed: bool = False        # ★追加
    confirmed_free_width: float = -1.0      # ★追加

    def update(self, candidate: DoorCandidate):
        self.position = candidate.position
        self.width = candidate.width
        self.yaw = candidate.yaw
        self.width_history.append(candidate.width)
        self.depth_history.append(candidate.depth_score)
        self.occupancy_history.append(candidate.occupancy_score)
        self.beyond_history.append(candidate.beyond_open)   
        self.horizontal_gap_history.append(candidate.horizontal_gap)
        if candidate.angle_diff >= 0:
            self.angle_history.append(candidate.angle_diff)
        self.hit_count += 1
        self.miss_count = 0
        # ★追加：widthの安定性を確認してから確定させる
        if self.hit_count >= 3:
            if len(self.width_history) >= 3:
                recent_widths = list(self.width_history)[-5:]
                width_std = float(np.std(recent_widths))
                if width_std < 0.20:   # 15cm以上バラついてたら「同じドア」と信用しない
                    self.confirmed = True
                else:
                    self.confirmed = False
            else:
                self.confirmed = True

    def mark_missed(self):
        self.miss_count += 1
        self.hit_count = max(0, self.hit_count - 1)
        if self.hit_count == 0:
            self.confirmed = False   # ★追加：完全にロストしたら確定も取り消す

    def classify(self, passable_thresh: float = 0.7) -> str:
        if len(self.beyond_history) < 3:
            return "unknown"

        recent_beyond = list(self.beyond_history)[-7:]
        votes = ["open" if b else "closed" for b in recent_beyond]

        n_open = votes.count("open")
        n_closed = votes.count("closed")

        if len(self.angle_history) >= 3:
            angles = list(self.angle_history)[-5:]
            angle_std = float(np.std(angles))
            angle_mean = float(np.mean(angles))
            if angle_std < 0.05 and angle_mean > 0.3:
                n_open += 2

        # ★追加：僅差(差が2票未満)のときは前回の状態を維持する(ヒステリシス)
        if abs(n_open - n_closed) < 2:
            return self.state

        self.state = "open" if n_open > n_closed else "closed"
        if self.state != 'open':
            self.passable_confirmed = False
            self.confirmed_free_width = -1.0

        return self.state
    
    def try_confirm_passability(self, dist_to_door: float, horizontal_gap: float,
                                  confirm_dist: float = 1.0):
        """ロボットがドアに十分近づいたら、通行可能幅を確定させる。
        free_widthは近距離でないと信頼できないため、遠距離では呼ばれても
        何もしない。"""
        if self.state != 'open':          # ★追加：closedやunknownなら何もしない
            return
        if dist_to_door <= confirm_dist:
            self.passable_confirmed = True
            self.confirmed_free_width = horizontal_gap

class DoorDetectorNode(Node):

    def __init__(self):
        super().__init__('door_detector')

        # --- パラメータ ---
        self.declare_parameter('max_detection_range', 4.0)
        self.declare_parameter('num_walls', 8)
        self.declare_parameter('ransac_iterations', 25)
        self.declare_parameter('ransac_inlier_thresh', 0.05)
        self.declare_parameter('min_wall_points', 25)
        self.declare_parameter('min_wall_length', 0.3)

        self.declare_parameter('door_width_min', 0.5)
        self.declare_parameter('door_width_max', 1.4)
        self.declare_parameter('door_angle_min', 0.25)   # 壁同士が「違う向き」とみなす角度[rad]

        # v4: ロボット幅60cm基準の通行可能幅しきい値
        self.declare_parameter('passable_width_thresh', 0.7)
        # v4: 枠検出/リーフ検出まわりのしきい値
        self.declare_parameter('frame_line_dist_thresh', 0.09)
        self.declare_parameter('frame_dir_sim_thresh', 0.90)
        self.declare_parameter('leaf_angle_thresh', 0.15)  # rad, これ未満は壁と同一とみなす

        self.declare_parameter('recess_min_depth', 0.02)
        self.declare_parameter('recess_max_depth', 0.20)   # 20cmまで拡張
        self.declare_parameter('recess_min_length', 0.3)
        self.declare_parameter('recess_max_length', 1.3)   # ドアとして不自然に長すぎないか

        # ★追加：奥行き検出・通行可能判定まわり
        self.declare_parameter('beyond_min_points', 20)          # 奥まで届いた点の最低数
        self.declare_parameter('beyond_margin', 0.5)            # 壁より何m以上遠ければ「奥」とみなすか
        self.declare_parameter('passability_confirm_dist', 1.0) # 通行可否を確定させる距離

        self.declare_parameter('track_match_dist', 0.5)
        self.declare_parameter('track_max_miss', 8)

        self.declare_parameter('self_filter_x', 0.50)
        self.declare_parameter('self_filter_y', 0.45)
        self.declare_parameter('self_filter_z_min', -0.15)
        self.declare_parameter('self_filter_z_max', 0.15)
        self.declare_parameter('wall_height_min', 0.05)     # 床からの高さ[m]
        self.declare_parameter('wall_height_max', 0.5)      # 床からの高さ[m]
        self.declare_parameter('knob_height_min', 0.80)     # 床からの高さ[m]
        self.declare_parameter('knob_height_max', 1.20)     # 床からの高さ[m]
        self.declare_parameter('floor_z_offset', -0.205)    # lidar_linkから見た床のz座標

        # v4.1: tf関連
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('sensor_frame', 'lidar_link')

        self._load_params()

        self.tracks: Dict[int, DoorTrack] = {}
        self.next_id = 0

        # v4.1: tf2 buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_points = self.create_subscription(
            PointCloud2, '/points', self.points_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/door_markers', 10)
        self.pub_info = self.create_publisher(String, '/door_info', 10)

        self.get_logger().info('DoorDetector v4.1 (frame-discontinuity + leaf-angle + tf2) started')

    def _load_params(self):
        g = self.get_parameter
        self.max_range          = g('max_detection_range').value
        self.num_walls           = g('num_walls').value
        self.ransac_iters        = g('ransac_iterations').value
        self.ransac_inlier_thresh = g('ransac_inlier_thresh').value
        self.min_wall_points     = g('min_wall_points').value
        self.min_wall_length     = g('min_wall_length').value

        self.door_width_min = g('door_width_min').value
        self.door_width_max = g('door_width_max').value
        self.door_angle_min = g('door_angle_min').value

        self.passable_width_thresh = g('passable_width_thresh').value
        self.frame_line_dist_thresh = g('frame_line_dist_thresh').value
        self.frame_dir_sim_thresh = g('frame_dir_sim_thresh').value
        self.leaf_angle_thresh = g('leaf_angle_thresh').value

        self.recess_min_depth  = g('recess_min_depth').value
        self.recess_max_depth  = g('recess_max_depth').value
        self.recess_min_length = g('recess_min_length').value
        self.recess_max_length = g('recess_max_length').value   

        self.beyond_min_points = g('beyond_min_points').value
        self.beyond_margin = g('beyond_margin').value
        self.passability_confirm_dist = g('passability_confirm_dist').value

        self.track_match_dist = g('track_match_dist').value
        self.track_max_miss   = g('track_max_miss').value

        self.self_filter_x = g('self_filter_x').value
        self.self_filter_y = g('self_filter_y').value
        self.self_filter_z_min = g('self_filter_z_min').value
        self.self_filter_z_max = g('self_filter_z_max').value
        self.wall_height_min = g('wall_height_min').value
        self.wall_height_max = g('wall_height_max').value
        self.knob_height_min = g('knob_height_min').value
        self.knob_height_max = g('knob_height_max').value
        self.floor_z_offset  = g('floor_z_offset').value

        self.map_frame = g('map_frame').value
        self.sensor_frame = g('sensor_frame').value

    # =====================================================================
    # メインコールバック
    # =====================================================================

    def points_callback(self, msg: PointCloud2):
        points = self._parse_pointcloud2(msg)
        if points is None or len(points) < 50:
            return

        points = self._remove_self(points)
        if len(points) < 50:
            return

        dist = np.linalg.norm(points[:, :2], axis=1)
        points = points[dist <= self.max_range]
        if len(points) < 50:
            return

        # 床からの相対高さに変換
        floor_relative_z = points[:, 2] - self.floor_z_offset

        # 壁検出用(床から0.05〜0.5m)
        height_points = points[
            (floor_relative_z >= self.wall_height_min) &
            (floor_relative_z <= self.wall_height_max)
        ]
        if len(height_points) < 50:
            return

        # ノブ検出用(床から0.8〜1.2m)
        knob_points = points[
            (floor_relative_z >= self.knob_height_min) &
            (floor_relative_z <= self.knob_height_max)
        ]

        walls = self._extract_walls(height_points)
        walls, recess_walls = self._merge_walls(walls)
        candidates = self._find_door_candidates(walls, points, recess_walls)

        # v4.1: 候補位置を lidar_link -> map に変換する
        candidates = self._transform_candidates_to_map(candidates, msg.header.stamp)

        self._update_tracks(candidates)
        self._prune_tracks()

        confirmed = [
            (tid, t)
            for tid, t in self.tracks.items()
            if t.confirmed and t.width_history
        ]

        # ★追加:どのdoorIDがどの座標・状態なのか一目で分かるようにする
        for tid, t in confirmed:
            self.get_logger().info(
                f"[track] Door{tid}: pos=({t.position[0]:.2f},{t.position[1]:.2f}) "
                f"width={t.width:.2f} state={t.state} hit_count={t.hit_count}"
            )
        # v4.8: ロボットの現在位置をtfから取得（通行可否の距離判定に使う）
        robot_pos = None
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.sensor_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            robot_pos = np.array([
                robot_transform.transform.translation.x,
                robot_transform.transform.translation.y
            ])
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            pass

        # points_callback 内
        for tid, t in confirmed:
            t.classify(passable_thresh=self.passable_width_thresh)
            if robot_pos is not None and t.horizontal_gap_history and t.state == 'open':   # ★ depth_history → horizontal_gap_history
                dist_now = float(np.linalg.norm(t.position - robot_pos))
                was_confirmed = t.passable_confirmed
                t.try_confirm_passability(
                    dist_now, t.horizontal_gap_history[-1],   # ★ free_widthの代わりにhorizontal_gapを渡す
                    confirm_dist=self.passability_confirm_dist
                )
                if t.passable_confirmed and not was_confirmed:
                    self.get_logger().info(
                        f"[debug] Door{tid} passable確定: dist={dist_now:.2f} "
                        f"horizontal_gap={t.confirmed_free_width:.2f} state={t.state}"   # ★ログ文言だけ更新
                    )
        self._publish_markers(confirmed)
        self._publish_info(confirmed)

        n_closed = sum(1 for _, t in confirmed if t.state == 'closed')
        n_open   = sum(1 for _, t in confirmed if t.state == 'open')
        self.get_logger().info(
            f'walls={len(walls)} candidates={len(candidates)} confirmed={len(confirmed)} '
            f'closed={n_closed} open={n_open}',
            throttle_duration_sec=2.0)

    # =====================================================================
    # v4.1: tf変換
    # =====================================================================

    def _transform_candidates_to_map(self, candidates: List[DoorCandidate], stamp) -> List[DoorCandidate]:
        if not candidates:
            return []

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.sensor_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f'tf変換取得失敗: {e}', throttle_duration_sec=2.0)
            return []

        transformed = []
        for c in candidates:
            pt = PointStamped()
            pt.header.frame_id = self.sensor_frame
            pt.header.stamp = stamp
            pt.point.x = float(c.position[0])
            pt.point.y = float(c.position[1])
            pt.point.z = 0.0

            pt_map = do_transform_point(pt, transform)
            c.position = np.array([pt_map.point.x, pt_map.point.y])
            self.get_logger().info(f"candidate map_pos=({c.position[0]:.3f},{c.position[1]:.3f})") 

            # yawもmapフレームに合わせて回転分を加算する
            q = transform.transform.rotation
            yaw_offset = 2.0 * np.arctan2(q.z, q.w)
            c.yaw = float(c.yaw + yaw_offset)

            transformed.append(c)

        return transformed

    # =====================================================================
    # 前処理
    # =====================================================================

    def _parse_pointcloud2(self, msg: PointCloud2) -> Optional[np.ndarray]:
        fields = {f.name: f.offset for f in msg.fields}
        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            return None
        point_step = msg.point_step
        data = msg.data
        n_points = msg.width * msg.height
        points = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            base = i * point_step
            points[i] = [
                struct.unpack_from('f', data, base + fields['x'])[0],
                struct.unpack_from('f', data, base + fields['y'])[0],
                struct.unpack_from('f', data, base + fields['z'])[0],
            ]
        valid = np.all(np.isfinite(points), axis=1)
        return points[valid]

    def _remove_self(self, points: np.ndarray) -> np.ndarray:
        mask = ~(
            (np.abs(points[:, 0]) < self.self_filter_x) &
            (np.abs(points[:, 1]) < self.self_filter_y) &
            (points[:, 2] > self.self_filter_z_min) &
            (points[:, 2] < self.self_filter_z_max)
        )
        return points[mask]

    # =====================================================================
    # 壁抽出 (効率化RANSAC、インデックスベースの除外)
    # =====================================================================

    def _extract_walls(self, points_3d: np.ndarray) -> List[Wall]:
        walls: List[Wall] = []
        remaining = points_3d[:, :2].copy()  # XYのみで処理

        for _ in range(self.num_walls):
            n_pts = len(remaining)
            if n_pts < self.min_wall_points:
                break

            best_mask = None
            best_score = 0

            for _ in range(self.ransac_iters):
                i1, i2 = np.random.choice(n_pts, 2, replace=False)
                p1 = remaining[i1]
                p2 = remaining[i2]
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                direction = direction / norm

                diffs = remaining - p1
                proj = diffs @ direction
                closest = p1 + np.outer(proj, direction)
                dist = np.linalg.norm(remaining - closest, axis=1)
                mask = dist < self.ransac_inlier_thresh
                score = int(np.sum(mask))

                if score > best_score:
                    best_score = score
                    best_mask = mask

            if best_mask is None or best_score < self.min_wall_points:
                break

            wall_pts = remaining[best_mask]
            wall = self._fit_wall(wall_pts)
            if wall is not None and wall.length >= self.min_wall_length:
                walls.append(wall)

            remaining = remaining[~best_mask]

        return walls

    def _merge_walls(self, walls):
        if len(walls) <= 1:
            return walls, []
        
        self.get_logger().info(f"[debug] merge対象の壁の本数: {len(walls)}")  # ★追加


        merged = []
        recess_candidates = []
        used = [False] * len(walls)

        for i in range(len(walls)):
            if used[i]:
                continue

            base = walls[i]
            cluster = [base]
            used[i] = True

            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue

                w = walls[j]

                # ① 方向が似ているか（平行か）
                dir_sim = abs(base.direction @ w.direction)
                if dir_sim < 0.95:
                    continue

                line_dist = abs(
                    np.cross(
                        base.direction,
                        w.start - base.start
                    )
                )

                # ★ここに追加
                self.get_logger().info(
                    f"[debug] dir_sim={dir_sim:.3f} line_dist={line_dist:.4f} w.length={w.length:.2f} "
                    f"(recess range: {self.recess_min_depth}-{self.recess_max_depth})"
                )

                # ★奥まり判定：2〜10cm程度なら「閉じたドア板」候補として記録
                if self.recess_min_depth <= line_dist <= self.recess_max_depth:
                    split_segments = self._split_recess_wall(w)
                    recess_candidates.extend(split_segments)
                    used[j] = True
                    continue

                if line_dist > 0.08:
                    continue

                # ② 距離が近いか（端点同士）
                d = min(
                    np.linalg.norm(base.start - w.start),
                    np.linalg.norm(base.start - w.end),
                    np.linalg.norm(base.end - w.start),
                    np.linalg.norm(base.end - w.end),
                )

                if d < 0.3:  # ←ここは調整値
                    cluster.append(w)
                    used[j] = True

            # ③ クラスタを1本にまとめる
            pts = np.vstack([w.points for w in cluster])
            merged_wall = self._fit_wall(pts)
            if merged_wall is not None:
                merged.append(merged_wall)

        return merged, recess_candidates
    
    def _split_recess_wall(self, w: Wall) -> List[Wall]:
        """奥まった1本の長い直線を、点群の密度が途切れる場所で
        複数の短い区画（ドア1枚ぶんずつ）に分割する。

        RANSACは直線からの距離しか見ておらず、直線に沿った連続性を
        考慮しないため、間に空白があっても同一直線上の点をまとめて
        1本の壁として抽出してしまうことがある（複数のドアが同じ
        奥まり量・同じ向きで並んでいる場合など）。
        _find_door_frames の射影+gap検出と同じ考え方で、点群を
        方向ベクトルに射影し、間隔が door_width_max を超える箇所で
        区画を分割し直す。
        """
        pts = w.points  # (N, 2)
        proj = (pts - w.start) @ w.direction
        order = np.argsort(proj)
        proj_sorted = proj[order]
        pts_sorted = pts[order]

        # 点と点の間隔が door_width_max を超えたら、そこで区画を分ける
        split_gap_thresh = self.door_width_max
        gaps = np.diff(proj_sorted)
        split_indices = np.where(gaps > split_gap_thresh)[0]

        segments = []
        start_idx = 0
        for idx in split_indices:
            segments.append(pts_sorted[start_idx:idx + 1])
            start_idx = idx + 1
        segments.append(pts_sorted[start_idx:])

        result = []
        for seg in segments:
            if len(seg) < 5:
                continue
            fit = self._fit_wall(seg)
            if fit is not None and self.recess_min_length <= fit.length <= self.recess_max_length:
                result.append(fit)

        self.get_logger().info(
            f"[debug] recess split: 元の長さ={w.length:.2f} -> {len(result)}個の区画 "
            f"(lengths={[round(f.length, 2) for f in result]})"
        )

        return result
    
    def _fit_wall(self, pts_2d: np.ndarray) -> Optional[Wall]:
        if len(pts_2d) < 2:
            return None
        center = pts_2d.mean(axis=0)
        cov = np.cov((pts_2d - center).T)
        if cov.ndim < 2:
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        proj = (pts_2d - center) @ direction
        start = center + proj.min() * direction
        end = center + proj.max() * direction
        length = float(proj.max() - proj.min())
        return Wall(
            points=pts_2d, start=start, end=end,
            direction=direction, length=length
        )

    def _angle_between(self, d1: np.ndarray, d2: np.ndarray) -> float:
        return float(np.arccos(np.clip(np.abs(d1 @ d2), 0.0, 1.0)))

    # =====================================================================
    # v4: ドア枠検出(同一直線上の壁の分断を探す) + リーフ検出 + 通行幅計算
    # =====================================================================

    def _find_door_candidates(self, walls: List[Wall], all_points_3d: np.ndarray,recess_walls: List[Wall] = None) -> List[DoorCandidate]:
        frames = self._find_door_frames(walls)
        self.get_logger().info(f"frames = {len(frames)}")
        frame_candidates = [self._classify_frame(f, all_points_3d) for f in frames]

        trusted_frame = [c for c in frame_candidates if c.angle_diff >= 0]
        untrusted_frame = [c for c in frame_candidates if c.angle_diff < 0]

        final_candidates = list(trusted_frame)  # 信頼できるframe候補は無条件で採用
        dedup_dist = 0.4

        # recess候補：信頼できるframe候補と近ければ捨てる(信頼できるframeを優先)
        filtered_recess = []
        if recess_walls:
            recess_candidates = self._recess_to_candidates(recess_walls)
            for rc in recess_candidates:
                near_trusted = any(
                    np.linalg.norm(rc.position - c.position) < dedup_dist
                    for c in trusted_frame
                )
                if near_trusted:
                    self.get_logger().info(
                        f"recess candidate重複除外: pos=({rc.position[0]:.2f},{rc.position[1]:.2f}) "
                        f"信頼できるframe候補と近すぎるため"
                    )
                else:
                    filtered_recess.append(rc)
        final_candidates.extend(filtered_recess)

        # ★追加：信頼できないframe候補は、recess候補と近ければ捨てる(recessを優先)
        for uc in untrusted_frame:
            near_recess = any(
                np.linalg.norm(uc.position - rc.position) < dedup_dist
                for rc in filtered_recess
            )
            if near_recess:
                self.get_logger().info(
                    f"未確定frame candidate除外: pos=({uc.position[0]:.2f},{uc.position[1]:.2f}) "
                    f"recess候補が既にあるため(leaf不明でbeyond_openのみの候補は不採用)"
                )
                continue

            # ★追加：近くにrecessが無くても、遠距離の未確定候補はそもそも信頼できないので捨てる
            dist_uc = float(np.linalg.norm(uc.position))
            if dist_uc > 1.5:
                self.get_logger().info(
                    f"未確定frame candidate除外: pos=({uc.position[0]:.2f},{uc.position[1]:.2f}) "
                    f"遠距離(dist={dist_uc:.2f})でleaf不明のため不採用"
                )
                continue

            final_candidates.append(uc)

        return final_candidates
    
    def _recess_to_candidates(self, recess_walls: List[Wall]) -> List[DoorCandidate]:
        """奥まった壁（＝閉じたドア板）を、隙間検出を経由せず直接 DoorCandidate に変換する。"""
        candidates = []
        for w in recess_walls:
            center = (w.start + w.end) / 2.0
            yaw = float(np.arctan2(w.direction[1], w.direction[0]))

            self.get_logger().info(
                f"recess candidate: length={w.length:.2f} center=({center[0]:.2f},{center[1]:.2f})"
            )

            candidates.append(DoorCandidate(
                position=center, width=w.length, yaw=yaw,
                gap_score=0.0, depth_score=0.0,
                height_score=1.0, occupancy_score=1.0,
                angle_diff=-1.0,
            ))
        return candidates

    def _find_door_frames(self, walls: List[Wall]) -> List[dict]:
        frames = []
        groups = self._group_collinear_walls(walls)

        for group in groups:
            pts = np.vstack([w.points for w in group])
            direction = group[0].direction
            base = group[0].start

            proj = (pts - base) @ direction
            proj_sorted = np.sort(proj)

            gaps = np.diff(proj_sorted)
            if len(gaps) == 0:
                continue

            for gap_idx in range(len(gaps)):
                gap_width = float(gaps[gap_idx])

                if not (self.door_width_min <= gap_width <= self.door_width_max):
                    continue

                # v4.2: gapの両端が「孤立点」ではなく、しっかりした壁の集団に
                # 支えられているかを確認する
                if not self._has_solid_support(proj_sorted, gap_idx):
                    continue

                p_start = base + proj_sorted[gap_idx] * direction
                p_end = base + proj_sorted[gap_idx + 1] * direction

                frames.append(dict(
                    edge_a=p_start,
                    edge_b=p_end,
                    frame_width=gap_width,
                    wall_yaw=float(np.arctan2(direction[1], direction[0])),
                    center=(p_start + p_end) / 2.0,
                ))

        return self._merge_frames(frames)

    def _has_solid_support(self, proj_sorted: np.ndarray, gap_idx: int,
                            window: float = 0.15, min_support: int = 4) -> bool:
        """gapの両端(手前側・奥側)それぞれに、windowの範囲内で
        min_support点以上の密な点群が存在するかを確認する。
        孤立した1〜2点だけがgapを作ってる場合はFalseを返す。"""

        left_edge = proj_sorted[gap_idx]
        right_edge = proj_sorted[gap_idx + 1]

        left_support = np.sum(
            (proj_sorted >= left_edge - window) & (proj_sorted <= left_edge)
        )
        right_support = np.sum(
            (proj_sorted >= right_edge) & (proj_sorted <= right_edge + window)
        )

        return left_support >= min_support and right_support >= min_support
    
    def _group_collinear_walls(self, walls: List[Wall]) -> List[List[Wall]]:
        """ほぼ同一直線上にある壁同士をグループ化する"""
        groups = []
        used = [False] * len(walls)
        for i, wa in enumerate(walls):
            if used[i]:
                continue
            group = [wa]
            used[i] = True
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                wb = walls[j]
                dir_sim = abs(wa.direction @ wb.direction)
                if dir_sim < self.frame_dir_sim_thresh:
                    continue
                line_dist = abs(np.cross(wa.direction, wb.start - wa.start))
                if line_dist > self.frame_line_dist_thresh:
                    continue
                group.append(wb)
                used[j] = True
            groups.append(group)
        return groups

    def _merge_frames(self, frames: List[dict]) -> List[dict]:
        if not frames:
            return []
        merged = []
        used = [False] * len(frames)
        for i, f in enumerate(frames):
            if used[i]:
                continue
            cluster = [f]
            used[i] = True
            for j in range(i + 1, len(frames)):
                if used[j]:
                    continue
                if np.linalg.norm(frames[j]['center'] - f['center']) < 0.3:
                    cluster.append(frames[j])
                    used[j] = True
            merged.append(dict(
                edge_a=cluster[0]['edge_a'],
                edge_b=cluster[0]['edge_b'],
                frame_width=float(np.mean([c['frame_width'] for c in cluster])),
                wall_yaw=float(np.mean([c['wall_yaw'] for c in cluster])),
                center=np.mean([c['center'] for c in cluster], axis=0),
            ))
        return merged
    
    def _reliability_adjusted_angle_thresh(self, dist_to_door: float) -> float:
        """距離が離れるほど、angle_diffが90度に近い(=壁の角の疑いがある)
        検出をより厳しく疑うための、距離依存の棄却しきい値を返す。
        近距離(1.5m以内)は緩め、遠距離(4m近く)は厳しめにする。"""
        near, far = 1.5, 4.0
        thresh_near, thresh_far = 1.45, 1.0

        if dist_to_door <= near:
            return thresh_near
        if dist_to_door >= far:
            return thresh_far

        ratio = (dist_to_door - near) / (far - near)
        return thresh_near + ratio * (thresh_far - thresh_near)

    def _reliability_adjusted_min_points(self, dist_to_door: float) -> int:
        """遠距離ほど点群密度が下がるので、beyond判定に必要な最低点数を
        距離に応じて下げる(ただし下限は設ける)。"""
        near, far = 1.5, 4.0
        pts_near, pts_far = self.beyond_min_points, max(8, self.beyond_min_points // 2)

        if dist_to_door <= near:
            return pts_near
        if dist_to_door >= far:
            return pts_far

        ratio = (dist_to_door - near) / (far - near)
        return int(round(pts_near + ratio * (pts_far - pts_near)))
    
    def _detect_depth_beyond_frame(self, frame: dict, points_3d: np.ndarray,
                                     wall_dist_estimate: float) -> bool:
        """枠の見える角度範囲の中に、壁までの距離より明らかに遠い点が
        あるかを調べる。扉の種類（開き戸／引き戸）を問わず、
        通路が空いているかどうかを判定できる指標。

        v4.9: 斜め視点への対応。ロボットが枠の正面ではなく斜め（廊下を
        並走しているときなど）から見ている場合、枠の「見た目の幅」は
        実際の frame_width より狭くなる（遠近法による圧縮）。
        これを補正しないと、見る角度の範囲を実際より広く取ってしまい、
        隣の枠や廊下奥の壁まで誤って拾ってしまう。
        枠の向き(wall_yaw)とロボットの視線方向(frame_angle)の差から
        視聴角度を求め、cosで見た目の幅を補正する。
        """
        xy = points_3d[:, :2]
        dist_from_sensor = np.linalg.norm(xy, axis=1)

        # ロボットから見た「枠の中心方向」の角度
        frame_angle = np.arctan2(frame['center'][1], frame['center'][0])

        # v4.9: 視聴角度を求める（枠の向きの法線方向と、ロボットの視線方向の差）
        # frame['wall_yaw'] は壁が伸びている方向なので、+pi/2 したものが壁の法線（開口部の正面方向）
        wall_normal = frame['wall_yaw'] + np.pi / 2.0
        viewing_angle = np.arctan2(
            np.sin(frame_angle - wall_normal),
            np.cos(frame_angle - wall_normal)
        )
        # cosが小さい（真横に近い）ほど見た目の幅が狭くなる。0除算防止に下限を設ける
        foreshorten = max(abs(np.cos(viewing_angle)), 0.15)
        apparent_width = frame['frame_width'] * foreshorten

        half_angle = np.arctan2(apparent_width * 0.5, max(wall_dist_estimate, 0.1))

        point_angles = np.arctan2(xy[:, 1], xy[:, 0])
        angle_diff_to_frame = np.abs(
            np.arctan2(np.sin(point_angles - frame_angle), np.cos(point_angles - frame_angle))
        )

        near_frame_mask = angle_diff_to_frame < half_angle
        beyond_wall_mask = dist_from_sensor > (wall_dist_estimate + self.beyond_margin)

        beyond_points = points_3d[near_frame_mask & beyond_wall_mask]
        required_points = self._reliability_adjusted_min_points(wall_dist_estimate)

        if len(beyond_points) >= required_points:
            self.get_logger().info(
                f"[debug] beyond points found: n={len(beyond_points)} "
                f"viewing_angle_deg={np.degrees(viewing_angle):.1f} foreshorten={foreshorten:.2f} "
                f"dists={[round(float(np.linalg.norm(p[:2])), 2) for p in beyond_points[:5]]}"
            )

        return len(beyond_points) >= required_points

    def _detect_leaf(self, frame: dict, points_3d: np.ndarray):
        frame_width = frame['frame_width']
        wall_yaw = frame['wall_yaw']
        wall_dir = np.array([np.cos(wall_yaw), np.sin(wall_yaw)])

        xy = points_3d[:, :2]

        best = None

        # edge_a側・edge_b側、それぞれ「ここが蝶番やった場合」を独立に試す
        for hinge_pt in (frame['edge_a'], frame['edge_b']):
            dist_to_hinge = np.linalg.norm(xy - hinge_pt, axis=1)

            # 壁の直線からの垂直距離(壁沿い・recessの点を除外するため)
            diffs = xy - hinge_pt
            perp_dist = np.abs(diffs[:, 0] * wall_dir[1] - diffs[:, 1] * wall_dir[0])

            # hingeからleaf_length分の範囲内、かつrecess_max_depthより浮いてる点だけ対象
            nearby_mask = (dist_to_hinge < frame_width * 1.3) & (perp_dist > self.recess_max_depth)
            nearby = points_3d[nearby_mask]

            if len(nearby) < 15:
                continue

            cluster = self._largest_cluster(nearby[:, :2], eps=0.08)
            if cluster is None or len(cluster) < 10:
                continue

            fit = self._fit_wall(cluster)
            if fit is None:
                continue

            if fit.length > frame_width * 1.5 or fit.length < 0.15:
                continue

            leaf_yaw = float(np.arctan2(fit.direction[1], fit.direction[0]))
            angle_diff = self._angle_between(fit.direction, wall_dir)

            if angle_diff < self.leaf_angle_thresh:
                continue

            candidate = dict(
                leaf_yaw=leaf_yaw, leaf_length=fit.length,
                angle_diff=angle_diff, leaf_centroid=cluster.mean(axis=0)
            )

            # 両端で候補が取れた場合は、より短い(=枠に収まってる)方を採用
            if best is None or fit.length < best['leaf_length']:
                best = candidate

        if best is None:
            self.get_logger().info("leaf reject: 両端どちらでも候補見つからず")
        else:
            self.get_logger().info(
                f"leaf angle_diff={best['angle_diff']:.2f} leaf_len={best['leaf_length']:.2f}"
            )

        return best
    
    def _largest_cluster(self, pts_2d: np.ndarray, eps: float = 0.15) -> Optional[np.ndarray]:
        """点群を距離epsで簡易クラスタリングし、最大のクラスタを返す"""
        n = len(pts_2d)
        if n == 0:
            return None

        visited = np.zeros(n, dtype=bool)
        best_cluster = None
        best_size = 0

        for i in range(n):
            if visited[i]:
                continue
            # 幅優先探索で塊を広げる
            stack = [i]
            visited[i] = True
            members = [i]
            while stack:
                idx = stack.pop()
                d = np.linalg.norm(pts_2d - pts_2d[idx], axis=1)
                neighbors = np.where((d < eps) & (~visited))[0]
                for nb in neighbors:
                    visited[nb] = True
                    stack.append(nb)
                    members.append(nb)

            if len(members) > best_size:
                best_size = len(members)
                best_cluster = pts_2d[members]

        return best_cluster

    def _classify_frame(self, frame: dict, points_3d: np.ndarray) -> DoorCandidate:
        dist_to_door = float(np.linalg.norm(frame['center']))

        # v4.8: 奥行き判定（メイン）。扉の種類を問わず使える
        beyond_open = self._detect_depth_beyond_frame(frame, points_3d, dist_to_door)

        leaf = self._detect_leaf(frame, points_3d)  # 角度判定（補助）は引き続き実行
        # ★追加：距離依存の急角度棄却
        if leaf is not None:
            angle_reject_thresh = self._reliability_adjusted_angle_thresh(dist_to_door)
            if leaf['angle_diff'] > angle_reject_thresh:
                self.get_logger().info(
                    f"leaf reject: 距離依存の急角度棄却 dist={dist_to_door:.2f} "
                    f"angle_diff={leaf['angle_diff']:.2f} thresh={angle_reject_thresh:.2f}"
                )
                leaf = None

        if leaf is None:
            return DoorCandidate(
                position=frame['center'], width=frame['frame_width'],
                yaw=frame['wall_yaw'], gap_score=0.0,
                depth_score=0.0, height_score=1.0, occupancy_score=1.0,
                angle_diff=-1.0, beyond_open=beyond_open,
            )

        if dist_to_door > 2.0 and leaf['angle_diff'] > 0.5:
            self.get_logger().info(
                f"leaf note: 遠距離+急角度 dist={dist_to_door:.2f} angle_diff={leaf['angle_diff']:.2f}"
            )

        hinge_pos = min(
            [frame['edge_a'], frame['edge_b']],
            key=lambda e: np.linalg.norm(e - leaf['leaf_centroid'])
        )
        far_edge = frame['edge_b'] if np.array_equal(hinge_pos, frame['edge_a']) else frame['edge_a']

        leaf_tip = hinge_pos + leaf['leaf_length'] * np.array(
            [np.cos(leaf['leaf_yaw']), np.sin(leaf['leaf_yaw'])]
        )
        free_width = float(np.linalg.norm(far_edge - leaf_tip))

        if free_width > frame['frame_width'] + 0.3:
            free_width = frame['frame_width']

        # ★追加：直進時の実効通行幅
        horizontal_gap = frame['frame_width'] - leaf['leaf_length'] * np.cos(leaf['angle_diff'])
        horizontal_gap = max(0.0, horizontal_gap)

        self.get_logger().info(
            f"free_width={free_width:.2f} horizontal_gap={horizontal_gap:.2f} dist={dist_to_door:.2f} "
            f"angle_diff={leaf['angle_diff']:.2f} beyond_open={beyond_open}"
        )

        return DoorCandidate(
            position=frame['center'], width=frame['frame_width'],
            yaw=frame['wall_yaw'], gap_score=free_width,
            depth_score=free_width, height_score=1.0, occupancy_score=0.0,
            angle_diff=leaf['angle_diff'], beyond_open=beyond_open,
            horizontal_gap=horizontal_gap,   # ★追加
        )
    # =====================================================================
    # トラッキング
    # =====================================================================

    def _update_tracks(self, candidates: List[DoorCandidate]):
        matched_track_ids = set()
        updated_this_cycle = set()   # ★追加

        for c in candidates:
            best_id = None
            best_dist = float('inf')
            for tid, t in self.tracks.items():
                if tid in updated_this_cycle:   # ★追加：今回もう更新済みなら除外
                    continue
                d = float(np.linalg.norm(c.position - t.position))
                if d < self.track_match_dist and d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id].update(c)
                matched_track_ids.add(best_id)
                updated_this_cycle.add(best_id)   # ★追加
            else:
                t = DoorTrack(position=c.position, width=c.width, yaw=c.yaw)
                t.width_history.append(c.width)
                t.depth_history.append(c.depth_score)
                t.occupancy_history.append(c.occupancy_score)
                t.beyond_history.append(c.beyond_open)
                if c.angle_diff >= 0:
                    t.angle_history.append(c.angle_diff)
                t.hit_count = 1
                self.tracks[self.next_id] = t
                matched_track_ids.add(self.next_id)
                updated_this_cycle.add(self.next_id)   # ★追加
                self.next_id += 1

        for tid, t in self.tracks.items():
            if tid not in matched_track_ids:
                t.mark_missed()

    def _prune_tracks(self):
        to_delete = [tid for tid, t in self.tracks.items() if t.miss_count > self.track_max_miss]
        for tid in to_delete:
            del self.tracks[tid]

    # =====================================================================
    # 可視化
    # =====================================================================

    def _publish_markers(self, confirmed: list):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for tid, t in confirmed:
            state = t.state
            half_w = t.width / 2.0
            dx = np.cos(t.yaw) * half_w
            dy = np.sin(t.yaw) * half_w

            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'door_frame'
            m.id = tid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08

            # ★変更：passable_confirmedを優先した色分け
            if state == 'open' and t.passable_confirmed:
                if t.confirmed_free_width >= self.passable_width_thresh:
                    m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0   # 青:通行確定
                else:
                    m.color.r, m.color.g, m.color.b = 1.0, 0.55, 0.0  # オレンジ:見た目は開いてるが幅不足
            elif state == 'open':
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            elif state == 'closed':
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0

            m.color.a = 1.0
            p1 = Point(); p1.x, p1.y = float(t.position[0] - dx), float(t.position[1] - dy)
            p2 = Point(); p2.x, p2.y = float(t.position[0] + dx), float(t.position[1] + dy)
            m.points = [p1, p2]
            marker_array.markers.append(m)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = now
            label.ns = 'door_label'
            label.id = tid
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(t.position[0])
            label.pose.position.y = float(t.position[1])
            label.pose.position.z = 0.5
            label.scale.z = 0.2
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0

            # ★変更：確定後はhorizontal_gapの実測値も表示
            if t.passable_confirmed:
                label.text = (
                    f"Door{tid}: {state.upper()}\n"
                    f"gap={t.confirmed_free_width:.2f}m"
                )
            else:
                label.text = f"Door{tid}: {state.upper()}\nw={t.width:.2f}m"
            marker_array.markers.append(label)

        self.pub_markers.publish(marker_array)

    def _publish_info(self, confirmed: list):
        if not confirmed:
            return
        lines = [
            f"Door{tid}: {t.state} pos=({t.position[0]:.2f},{t.position[1]:.2f}) width={t.width:.2f}m"
            for tid, t in confirmed
        ]
        msg = String()
        msg.data = '\n'.join(lines)
        self.pub_info.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()