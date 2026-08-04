#!/usr/bin/env python3
"""
door_detector_node.py (v3)

設計方針:
1. RANSACで壁(直線)を複数抽出する
2. 壁の両端点を使って、隣接する壁同士の隙間(ギャップ)を計算する
   (壁の重心同士ではなく、端点同士の距離を見ることで精度を上げる)
3. ギャップがドア幅相当(0.5~1.3m)で、かつ壁の向きが大きく異なる
   (廊下の延長ではなくドアらしい)組み合わせだけを候補とする
4. 候補位置の奥に、周辺の壁よりも遠い点があるかどうかも見て
   (ドアが開いて奥の部屋が見えているか)、判定の確からしさを補強する
5. トラッキング: 新規候補は仮登録とし、複数フレーム連続で
   観測されたものだけ正式なドアとして扱う。
   一定フレーム観測が無ければ自動的に削除する。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque


@dataclass
class Wall:
    points: np.ndarray      # この壁を構成する点群(N,3)
    start: np.ndarray       # 端点1 (XY)
    end: np.ndarray         # 端点2 (XY)
    direction: np.ndarray   # 単位方向ベクトル(XY)
    length: float


@dataclass
class DoorCandidate:
    position: np.ndarray    # XY
    width: float
    yaw: float
    gap_score: float        # ギャップが大きいほど高スコア(open寄り)
    depth_score: float      # 奥が見えるほど高スコア(open寄り)
    height_score: float
    occupancy_score: float
    total_score: float = 0.0


@dataclass
class DoorTrack:
    position: np.ndarray
    width: float
    yaw: float
    confirmed: bool = False
    hit_count: int = 0
    miss_count: int = 0
    width_history: deque = field(default_factory=lambda: deque(maxlen=8))
    depth_history: deque = field(default_factory=lambda: deque(maxlen=8))
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

    def classify(self, closed_w: float, open_w: float) -> str:

        if len(self.width_history) < 2:
            return "unknown"

        # 幅
        avg_width = float(np.mean(list(self.width_history)[-5:]))

        # Depth
        avg_depth = float(np.mean(self.depth_history))

        # Occupancy
        avg_occ = float(np.mean(self.occupancy_history))

        # -------------------------
        # 各特徴量を0～1に正規化
        # -------------------------

        width_score = np.clip(
            (avg_width - closed_w) / (open_w - closed_w),
            0.0,
            1.0
        )

        depth_score = np.clip(
            avg_depth / 0.20,      # 0.20m以上奥が見えれば十分
            0.0,
            1.0
        )

        occ_score = 1.0 - np.clip(avg_occ, 0.0, 1.0)

        # -------------------------
        # 総合スコア
        # -------------------------

        score = (
            0.30 * width_score +
            0.40 * depth_score +
            0.30 * occ_score
        )

        if score < 0.35:
            new_state = "closed"
        elif score < 0.65:
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

        self.declare_parameter('closed_width_thresh', 0.20)
        self.declare_parameter('open_width_thresh', 0.65)

        self.declare_parameter('track_match_dist', 0.5)
        self.declare_parameter('track_max_miss', 8)

        self.declare_parameter('self_filter_x', 0.50)
        self.declare_parameter('self_filter_y', 0.45)
        self.declare_parameter('self_filter_z_min', -0.15)
        self.declare_parameter('self_filter_z_max', 0.15)
        self.declare_parameter('wall_height_min', -0.1)
        self.declare_parameter('wall_height_max', 0.3)

        self._load_params()

        self.tracks: Dict[int, DoorTrack] = {}
        self.next_id = 0

        self.sub_points = self.create_subscription(
            PointCloud2, '/points', self.points_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/door_markers', 10)
        self.pub_info = self.create_publisher(String, '/door_info', 10)

        self.get_logger().info('DoorDetector v3 (wall-pair + depth) started')

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

        self.closed_width_thresh = g('closed_width_thresh').value
        self.open_width_thresh   = g('open_width_thresh').value

        self.track_match_dist = g('track_match_dist').value
        self.track_max_miss   = g('track_max_miss').value

        self.self_filter_x = g('self_filter_x').value
        self.self_filter_y = g('self_filter_y').value
        self.self_filter_z_min = g('self_filter_z_min').value
        self.self_filter_z_max = g('self_filter_z_max').value
        self.wall_height_min = g('wall_height_min').value
        self.wall_height_max = g('wall_height_max').value

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

        self._update_tracks(candidates)
        self._prune_tracks()

        confirmed = [
            (tid, t)
            for tid, t in self.tracks.items()
            if t.confirmed and t.width_history
        ]
        for tid, t in confirmed:
            t.classify(self.closed_width_thresh, self.open_width_thresh)

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
                        w.start-base.start
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

    # =====================================================================
    # ドア候補検出: 壁の端点同士のペアから、ドアらしいギャップを探す
    # =====================================================================

    def _find_door_candidates(self, walls: List[Wall], all_points_3d: np.ndarray) -> List[DoorCandidate]:
        candidates = []
        n = len(walls)
        self.get_logger().info(
            f"walls = {len(walls)}"
        )

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                wa = walls[i]
                wb = walls[j]

                # wa.end と wb.start の組み合わせで最も近いペアを探す
                # (4通りの端点組み合わせから最小距離のものを採用)
                pairs = [
                    (wa.end, wb.start), (wa.end, wb.end),
                    (wa.start, wb.start), (wa.start, wb.end),
                ]
                best_pair = min(pairs, key=lambda p: np.linalg.norm(p[0] - p[1]))
                pa, pb = best_pair
                gap = float(np.linalg.norm(pa - pb))
                angle_diff = self._angle_between(wa.direction, wb.direction)

                # 壁の向きが変すぎるものは除外
                if angle_diff > np.pi / 2 - self.door_angle_min:
                    continue

                vec = pb - pa
                dir_a = wa.direction

                along = abs(np.dot(vec, dir_a))
                cross = np.linalg.norm(vec - along * dir_a)

                # 横にズレすぎてるものは除外
                if cross > 0.2:
                    continue

                if gap<0.6:
                    continue

                if not (self.door_width_min <= gap <= self.door_width_max):
                    self.get_logger().info(f"Rejected: gap width={gap:.2f}")
                    continue

                if not (self.door_width_min <= gap <= self.door_width_max):
                    continue

                angle_diff = self._angle_between(wa.direction, wb.direction)
                # 同じ向き(廊下の延長線上)すぎるものは除外しない方が良いケースもあるが、
                # 全く違う壁(直交する壁)同士は除外する
                if angle_diff > np.pi / 2 - self.door_angle_min:
                    continue

                center = (pa + pb) / 2.0
                yaw = float(np.arctan2(wa.direction[1], wa.direction[0]))

                depth_score = self._compute_depth_score(center, yaw, gap, all_points_3d)
                height_score = self._compute_height_score(center, all_points_3d)
                occupancy = self._compute_occupancy(center, yaw, gap, all_points_3d)
                
                if occupancy > 0.3:
                    self.get_logger().info("Rejected: occupancy")
                    continue

                self.get_logger().info(
                    f"depth_score={depth_score:.3f}"
                )

                if depth_score < 0.15:
                    self.get_logger().info(f"Rejected: depth={depth_score:.2f}")
                    continue

                if height_score < 0.6:
                    self.get_logger().info(f"Rejected: height={height_score:.2f}")
                    continue

                self.get_logger().info(
                    f"gap={gap:.2f}, "
                    f"depth={depth_score:.2f}, "
                    f"height={height_score:.2f}, "
                    f"occ={occupancy:.2f}"
                )

                candidates.append(DoorCandidate(
                    position=center,
                    width=gap,
                    yaw=yaw,
                    gap_score=gap,
                    depth_score=depth_score,
                    height_score=height_score,
                    occupancy_score=occupancy,
                ))
        candidates = self._merge_candidates(candidates)

        for c in candidates:
            c.total_score = self._compute_door_score(c)
            self.get_logger().info(
                f"width={c.width:.2f}, "
                f"depth={c.depth_score:.2f}, "
                f"height={c.height_score:.2f}, "
                f"occ={c.occupancy_score:.2f}, "
                f"score={c.total_score:.2f}"
            )

        candidates = [c for c in candidates if c.total_score > 0.5]
        self.get_logger().info(
            f"candidates(after filter) = {len(candidates)}"
        )
        return candidates

    def _angle_between(self, d1: np.ndarray, d2: np.ndarray) -> float:
        return float(np.arccos(np.clip(np.abs(d1 @ d2), 0.0, 1.0)))

    def _compute_depth_score(self, position, yaw, width, points_3d):
        profile = self._compute_gap_profile(position, yaw, width, points_3d)
        self.get_logger().info(
            f"gap_profile nearest={profile['nearest']:.2f} density_near={profile['density_near']}"
        )
        # 遠くまで何も無いほどopen寄り
        return float(np.clip(profile['nearest'] / 1.0, 0.0, 1.0))
    
    def _compute_gap_profile(self, position, yaw, width, points_3d):
        """ドア開口部の奥行き方向に、最初に何かにぶつかる距離とその密度を返す"""
        yaw2 = yaw + np.pi / 2
        xy = points_3d[:, :2]
        c, s = np.cos(-yaw2), np.sin(-yaw2)
        R = np.array([[c, -s], [s, c]])
        local = (xy - position) @ R.T

        mask = (
            (np.abs(local[:, 0]) < width / 2 * 0.9) &
            (local[:, 1] > 0.0) &
            (local[:, 1] < 2.0)
        )
        depths = local[mask][:, 1]

        if len(depths) == 0:
            return dict(nearest=2.0, density_near=0)

        nearest = float(np.min(depths))
        density_near = int(np.sum(depths < 0.3))

        return dict(nearest=nearest, density_near=density_near)

    def _compute_height_score(self, position, points):
        xy = points[:, :2]
        z = points[:, 2]

        # ドア周辺だけ取る
        dist = np.linalg.norm(xy - position, axis=1)
        local = points[dist < 0.4]

        if len(local) < 20:
            return 0.0

        z_local = local[:, 2]

        # 高さをビン分割（重要）
        bins = np.linspace(-0.2, 1.8, 8)
        hist, _ = np.histogram(z_local, bins=bins)

        hist = hist / (np.max(hist) + 1e-6)

        # 下から上まで“空いてるか”
        free_ratio = np.sum(hist < 0.2) / len(hist)

        # 床付近が詰まってるならロッカー疑い
        floor_block = hist[0] > 0.5

        if floor_block:
            return 0.0

        return float(free_ratio)
    
    def _compute_occupancy(self, position, yaw, width, points):
        profile = self._compute_gap_profile(position, yaw, width, points)
        # 手前0.3m以内に点が密集してるほどclosed寄り
        return float(np.clip(profile['density_near'] / 30.0, 0.0, 1.0))
    
    def _merge_candidates(self, candidates: List[DoorCandidate]) -> List[DoorCandidate]:
        if not candidates:
            return []
        merged = []
        used = [False] * len(candidates)
        for i, c in enumerate(candidates):
            if used[i]:
                continue
            cluster = [c]
            used[i] = True
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                if np.linalg.norm(candidates[j].position - c.position) < 0.3:
                    cluster.append(candidates[j])
                    used[j] = True
            avg_pos = np.mean([cc.position for cc in cluster], axis=0)
            avg_width = float(np.mean([cc.width for cc in cluster]))
            avg_yaw = float(np.mean([cc.yaw for cc in cluster]))
            avg_depth = float(np.mean([cc.depth_score for cc in cluster]))
            avg_occ = float(np.mean([cc.occupancy_score for cc in cluster]))
            merged.append(DoorCandidate(
                position=avg_pos,
                width=avg_width,
                yaw=avg_yaw,
                gap_score=avg_width,
                depth_score=avg_depth,
                height_score=float(np.mean([cc.height_score for cc in cluster])),
                occupancy_score=avg_occ,
                total_score=0.0
            ))
        return merged
    
    def _compute_door_score(self, c):

        ideal_width = 0.9

        gap = np.exp(
            -((c.width-ideal_width)**2)/(2*0.15**2)
        )
        depth = np.clip(c.depth_score / 0.7, 0.0, 1.0)

        height = np.clip(c.height_score, 0.0, 1.0)
        occ = 1.0 - np.clip(c.occupancy_score,0,1)

        return (0.30 * gap + 0.30 * depth + 0.20 * height + 0.20*occ)

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
        delete_all.header.frame_id = 'lidar_link'
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for tid, t in confirmed:
            state = t.state
            half_w = t.width / 2.0
            dx = np.cos(t.yaw) * half_w
            dy = np.sin(t.yaw) * half_w

            m = Marker()
            m.header.frame_id = 'lidar_link'
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
            label.header.frame_id = 'lidar_link'
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
