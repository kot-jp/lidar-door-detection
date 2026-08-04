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
   open / semi-open / closed に分類する。
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
    total_score: float = 0.0


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
    state: str = 'unknown'

    def update(self, candidate: DoorCandidate):
        self.position = candidate.position
        self.width = candidate.width
        self.yaw = candidate.yaw
        self.width_history.append(candidate.width)
        self.depth_history.append(candidate.depth_score)
        self.occupancy_history.append(candidate.occupancy_score)
        self.hit_count += 1
        self.miss_count = 0
        if self.hit_count >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.miss_count += 1
        self.hit_count = max(0, self.hit_count - 1)

    def classify(self, passable_thresh: float = 0.7) -> str:
        """
        depth_history には free_width(実際に通れる幅)が積まれている。
        - free_width が ほぼ0        -> closed (リーフが検出できていない=閉まっている)
        - free_width < passable_thresh -> semi-open (開いてはいるがロボットは通れない)
        - free_width >= passable_thresh -> open (通行可能)
        """
        if len(self.depth_history) < 2:
            return "unknown"

        recent = list(self.depth_history)[-5:]
        avg_free_width = float(np.mean(recent))

        # v4.4: 直近フレームの中で、リーフが検出できた(free_width>0)割合を見る。
        # 斜め視点などで検出が乏しい場合、安易にclosedへ倒さず直前の状態を維持する
        detected_ratio = sum(1 for w in recent if w > 0.05) / len(recent)

        if detected_ratio < 0.3 and self.state in ("open", "semi-open"):
            return self.state

        if avg_free_width < 0.05:
            new_state = "closed"
        elif avg_free_width < passable_thresh:
            new_state = "semi-open"
        else:
            new_state = "open"

        # ヒステリシス
        if self.state == "closed" and new_state == "open":
            new_state = "semi-open"
        if self.state == "open" and new_state == "closed":
            new_state = "semi-open"

        self.state = new_state
        return new_state

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

        self.declare_parameter('door_width_min', 0.4)
        self.declare_parameter('door_width_max', 1.3)
        self.declare_parameter('door_angle_min', 0.25)   # 壁同士が「違う向き」とみなす角度[rad]

        # v4: ロボット幅60cm基準の通行可能幅しきい値
        self.declare_parameter('passable_width_thresh', 0.7)
        # v4: 枠検出/リーフ検出まわりのしきい値
        self.declare_parameter('frame_line_dist_thresh', 0.10)
        self.declare_parameter('frame_dir_sim_thresh', 0.90)
        self.declare_parameter('leaf_angle_thresh', 0.15)  # rad, これ未満は壁と同一とみなす

        self.declare_parameter('track_match_dist', 0.5)
        self.declare_parameter('track_max_miss', 8)

        self.declare_parameter('self_filter_x', 0.50)
        self.declare_parameter('self_filter_y', 0.45)
        self.declare_parameter('self_filter_z_min', -0.15)
        self.declare_parameter('self_filter_z_max', 0.15)
        self.declare_parameter('wall_height_min', -0.1)
        self.declare_parameter('wall_height_max', 0.3)

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

        self.track_match_dist = g('track_match_dist').value
        self.track_max_miss   = g('track_max_miss').value

        self.self_filter_x = g('self_filter_x').value
        self.self_filter_y = g('self_filter_y').value
        self.self_filter_z_min = g('self_filter_z_min').value
        self.self_filter_z_max = g('self_filter_z_max').value
        self.wall_height_min = g('wall_height_min').value
        self.wall_height_max = g('wall_height_max').value

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

        height_points = points[
            (points[:, 2] >= self.wall_height_min) &
            (points[:, 2] <= self.wall_height_max)
        ]
        if len(height_points) < 50:
            return

        walls = self._extract_walls(height_points)
        walls = self._merge_walls(walls)
        candidates = self._find_door_candidates(walls, points)

        # v4.1: 候補位置を lidar_link -> map に変換する
        candidates = self._transform_candidates_to_map(candidates, msg.header.stamp)

        self._update_tracks(candidates)
        self._prune_tracks()

        confirmed = [
            (tid, t)
            for tid, t in self.tracks.items()
            if t.confirmed and t.width_history
        ]
        for tid, t in confirmed:
            t.classify(passable_thresh=self.passable_width_thresh)

        self._publish_markers(confirmed)
        self._publish_info(confirmed)

        n_closed = sum(1 for _, t in confirmed if t.state == 'closed')
        n_semi   = sum(1 for _, t in confirmed if t.state == 'semi-open')
        n_open   = sum(1 for _, t in confirmed if t.state == 'open')
        self.get_logger().info(
            f'walls={len(walls)} candidates={len(candidates)} confirmed={len(confirmed)} '
            f'closed={n_closed} semi-open={n_semi} open={n_open}',
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
            return walls

        merged = []
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

        return merged

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

    def _find_door_candidates(self, walls: List[Wall], all_points_3d: np.ndarray) -> List[DoorCandidate]:
        frames = self._find_door_frames(walls)
        self.get_logger().info(f"frames = {len(frames)}")
        candidates = [self._classify_frame(f, all_points_3d) for f in frames]
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

    def _detect_leaf(self, frame: dict, points_3d: np.ndarray):
        center = frame['center']
        frame_width = frame['frame_width']
        wall_yaw = frame['wall_yaw']

        xy = points_3d[:, :2]
        dist = np.linalg.norm(xy - center, axis=1)
        nearby = points_3d[dist < frame_width * 1.2]

        if len(nearby) < 15:
            self.get_logger().info(f"leaf reject: nearby点数不足 n={len(nearby)}")
            return None

        # v4.2: 探索範囲内の点群を、近接クラスタに分割し、
        # 最大のクラスタだけをリーフ候補として使う
        # (隣接壁の点が紛れ込んで直線フィットが破綻するのを防ぐ)
        cluster = self._largest_cluster(nearby[:, :2], eps=0.15)
        if cluster is None or len(cluster) < 10:
            self.get_logger().info("leaf reject: クラスタ不足")
            return None

        fit = self._fit_wall(cluster)
        if fit is None:
            self.get_logger().info("leaf reject: fit失敗")
            return None

        if fit.length > frame_width * 1.5 or fit.length < 0.15:
            self.get_logger().info(
                f"leaf reject: 長さ不正 fit.length={fit.length:.2f} frame_width={frame_width:.2f}"
            )
            return None

        leaf_yaw = float(np.arctan2(fit.direction[1], fit.direction[0]))
        wall_dir = np.array([np.cos(wall_yaw), np.sin(wall_yaw)])
        angle_diff = self._angle_between(fit.direction, wall_dir)

        self.get_logger().info(f"leaf angle_diff={angle_diff:.2f} leaf_len={fit.length:.2f}")

        if angle_diff < self.leaf_angle_thresh:
            self.get_logger().info(f"leaf reject: 角度不足 angle_diff={angle_diff:.2f}")
            return None

        leaf_centroid = cluster.mean(axis=0)

        return dict(
            leaf_yaw=leaf_yaw, leaf_length=fit.length,
            angle_diff=angle_diff, leaf_centroid=leaf_centroid
        )

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
        leaf = self._detect_leaf(frame, points_3d)

        if leaf is None:
            return DoorCandidate(
                position=frame['center'], width=frame['frame_width'],
                yaw=frame['wall_yaw'], gap_score=0.0,
                depth_score=0.0, height_score=1.0, occupancy_score=1.0,
            )

        # 蝶番はリーフ点群の重心に近い方の端点(安定した基準)
        hinge_pos = min(
            [frame['edge_a'], frame['edge_b']],
            key=lambda e: np.linalg.norm(e - leaf['leaf_centroid'])
        )
        far_edge = frame['edge_b'] if np.array_equal(hinge_pos, frame['edge_a']) else frame['edge_a']

        leaf_tip = hinge_pos + leaf['leaf_length'] * np.array(
            [np.cos(leaf['leaf_yaw']), np.sin(leaf['leaf_yaw'])]
        )
        free_width = float(np.linalg.norm(far_edge - leaf_tip))

        # v4.3: 通行幅が枠幅を超えるのは物理的にあり得ない。
        # 蝶番の誤推定や隣接壁の混入が疑われるため、closed扱いとする
        if free_width > frame['frame_width'] + 0.3:
            self.get_logger().info(
                f"leaf reject: free_width異常 free_width={free_width:.2f} frame_width={frame['frame_width']:.2f}"
            )
            return DoorCandidate(
                position=frame['center'], width=frame['frame_width'],
                yaw=frame['wall_yaw'], gap_score=0.0,
                depth_score=0.0, height_score=1.0, occupancy_score=1.0,
            )

        self.get_logger().info(f"free_width={free_width:.2f}")

        return DoorCandidate(
            position=frame['center'], width=frame['frame_width'],
            yaw=frame['wall_yaw'], gap_score=free_width,
            depth_score=free_width, height_score=1.0, occupancy_score=0.0,
        )
    # =====================================================================
    # トラッキング
    # =====================================================================

    def _update_tracks(self, candidates: List[DoorCandidate]):
        matched_track_ids = set()

        for c in candidates:
            best_id = None
            best_dist = float('inf')
            for tid, t in self.tracks.items():
                d = float(np.linalg.norm(c.position - t.position))
                if d < self.track_match_dist and d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id].update(c)
                matched_track_ids.add(best_id)
            else:
                t = DoorTrack(position=c.position, width=c.width, yaw=c.yaw)
                t.width_history.append(c.width)
                t.depth_history.append(c.depth_score)
                t.occupancy_history.append(c.occupancy_score)
                t.hit_count = 1
                self.tracks[self.next_id] = t
                matched_track_ids.add(self.next_id)
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
            if state == 'open':
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            elif state == 'semi-open':
                m.color.r, m.color.g, m.color.b = 1.0, 0.6, 0.0
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