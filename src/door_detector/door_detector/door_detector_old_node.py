#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque
from sklearn.cluster import DBSCAN
import struct


@dataclass
class Segment:
    points: np.ndarray
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    length: float
    fit_error: float


@dataclass
class DoorCandidate:
    position: np.ndarray
    width: float
    yaw: float
    state: str
    confidence: float


@dataclass
class StableDoor:
    position: np.ndarray
    width: float
    yaw: float
    state: str
    history: deque = field(default_factory=lambda: deque(maxlen=7))

    def update(self, state: str):
        self.history.append(state)

    def stable_state(self) -> str:
        if len(self.history) < 2:
            return 'unknown'
        open_count      = self.history.count('open')
        closed_count    = self.history.count('closed')
        semi_open_count = self.history.count('semi-open')
        if open_count >= 3:
            return 'open'
        if closed_count >= 3:
            return 'closed'
        if semi_open_count >= 3:
            return 'semi-open'
        return 'unknown'


class DoorDetectorNode(Node):

    def __init__(self):
        super().__init__('door_detector')

        self.declare_parameter('door_width_min', 0.6)
        self.declare_parameter('door_width_max', 1.2)
        self.declare_parameter('wall_min_length', 1.0)      # 「壁」として確定する最小長さ
        self.declare_parameter('min_segment_points', 3)
        self.declare_parameter('point_gap_threshold', 0.05)
        self.declare_parameter('line_fit_threshold', 0.08)
        self.declare_parameter('max_detection_range', 4.0)
        self.declare_parameter('collinear_angle_threshold', 0.15)
        self.declare_parameter('open_width_threshold', 0.65)
        self.declare_parameter('wall_to_door_gap_max', 0.6)  # 壁とドア板の接続許容ギャップ
        self.declare_parameter('position_tolerance', 0.3)
        self.declare_parameter('self_filter_x', 0.50)
        self.declare_parameter('self_filter_y', 0.45)
        self.declare_parameter('self_filter_z_min', -0.15)
        self.declare_parameter('self_filter_z_max', 0.15)
        self.declare_parameter('slice_z_min', -0.3)
        self.declare_parameter('slice_z_max',  0.3)
        self.declare_parameter('floor_check_z', -0.25)
        self.declare_parameter('floor_check_tolerance', 0.15)

        self._load_params()

        self.stable_doors: Dict[int, StableDoor] = {}
        self.next_door_id = 0

        self.sub_points = self.create_subscription(
            PointCloud2, '/points', self.points_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/door_markers', 10)
        self.pub_info = self.create_publisher(String, '/door_info', 10)
        self.get_logger().info('DoorDetector 3D (sequential wall-door scan) started')

    def _load_params(self):
        self.door_width_min   = self.get_parameter('door_width_min').value
        self.door_width_max   = self.get_parameter('door_width_max').value
        self.wall_min_len     = self.get_parameter('wall_min_length').value
        self.min_seg_pts      = self.get_parameter('min_segment_points').value
        self.gap_thresh       = self.get_parameter('point_gap_threshold').value
        self.fit_thresh       = self.get_parameter('line_fit_threshold').value
        self.max_range        = self.get_parameter('max_detection_range').value
        self.collinear_angle_thresh = self.get_parameter('collinear_angle_threshold').value
        self.open_width_thresh = self.get_parameter('open_width_threshold').value
        self.wall_door_gap_max = self.get_parameter('wall_to_door_gap_max').value
        self.pos_tolerance    = self.get_parameter('position_tolerance').value
        self.self_filter_x    = self.get_parameter('self_filter_x').value
        self.self_filter_y    = self.get_parameter('self_filter_y').value
        self.self_filter_z_min = self.get_parameter('self_filter_z_min').value
        self.self_filter_z_max = self.get_parameter('self_filter_z_max').value
        self.slice_z_min      = self.get_parameter('slice_z_min').value
        self.slice_z_max      = self.get_parameter('slice_z_max').value
        self.floor_check_z    = self.get_parameter('floor_check_z').value
        self.floor_check_tol  = self.get_parameter('floor_check_tolerance').value

        self._raw_points = None

    def points_callback(self, msg: PointCloud2):
        points = self._parse_pointcloud2(msg)
        if points is None or len(points) < 10:
            return

        points = self._remove_self(points)
        dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        points = points[dist <= self.max_range]
        if len(points) < 10:
            return

        self._raw_points = points

        if len(points) < 50:
            return

        # ★ここが重要（zスライス廃止）
        segments = self._cluster_segments(points[:, :2])
        segments = self._sort_segments(segments)
        segments = self._merge_collinear_segments(segments)

        if len(segments) < 2:
            return

        all_candidates = self._scan_sequential(segments)

        all_candidates = self._merge_candidates(all_candidates)

        # 床チェック
        all_candidates = [
            c for c in all_candidates
            if c.state != 'closed' or self._check_floor_continuity(c)
        ]

        self._update_stable_doors(all_candidates)

        stable = [
            (did, d) for did, d in self.stable_doors.items()
            if d.stable_state() != 'unknown'
        ]

        self._publish_markers(stable)
        self._publish_info(stable)

        n_closed = sum(1 for c in all_candidates if c.state == 'closed')
        n_semi   = sum(1 for c in all_candidates if c.state == 'semi-open')
        n_open   = sum(1 for c in all_candidates if c.state == 'open')

        self.get_logger().info(
            f'total_segs={len(segments)} candidates={len(all_candidates)} '
            f'closed={n_closed} semi-open={n_semi} open={n_open} stable={len(stable)}',
            throttle_duration_sec=2.0
        )
        
    # =========================================================
    # 順次スキャン方式：
    # セグメントを順番に見ていき、「壁」を確定したら、
    # その直後のセグメントを「ドア候補」として評価する。
    # 次のセグメントが壁かどうかは問わない(前の壁だけで判定可能)。
    # =========================================================

    def _scan_sequential(self, segments: List[Segment]) -> List[DoorCandidate]:

        candidates = []

        # =========================
        # ① 壁→ドア→壁 検出
        # =========================
        for i in range(len(segments) - 2):

            left = segments[i]
            door = segments[i + 1]
            right = segments[i + 2]

            # 壁の最低長さ
            if left.length < self.wall_min_len:
                continue
            if right.length < self.wall_min_len:
                continue

            # ドアサイズチェック
            if not (self.door_width_min <= door.length <= self.door_width_max):
                continue

            # 左右壁の平行性
            wall_angle = self._angle_between(
                left.direction,
                right.direction
            )
            if wall_angle > np.deg2rad(10):
                continue

            # 壁との接続ギャップ
            gap_left = float(np.linalg.norm(door.start - left.end))
            gap_right = float(np.linalg.norm(right.start - door.end))

            if gap_left > self.wall_door_gap_max:
                continue
            if gap_right > self.wall_door_gap_max:
                continue

            # =========================
            # 開口幅（重要修正：中心点使用）
            # =========================
            door_center = (door.start + door.end) / 2

            dist_left = abs(
                (door_center - left.end)
                @ left.normal
            )

            dist_right = abs(
                (door_center - right.start)
                @ right.normal
            )

            opening = (dist_left + dist_right) / 2.0

            # =========================
            # 状態判定
            # =========================
            if opening < 0.15:
                state = "closed"
            elif opening < self.open_width_thresh:
                state = "semi-open"
            else:
                state = "open"

            # yaw安定化
            yaw = float(np.arctan2(
                door.direction[1],
                door.direction[0]
            ))

            # =========================
            # confidence（改善版）
            # =========================
            confidence = 1.0

            confidence -= gap_left * 0.5
            confidence -= gap_right * 0.5

            confidence -= wall_angle * 0.3

            confidence = float(np.clip(confidence, 0.0, 1.0))
            if confidence < 0.45:
                continue

            candidates.append(DoorCandidate(
                position=door_center,
                width=door.length,
                yaw=yaw,
                state=state,
                confidence=confidence
            ))

        # =========================
        # ② ギャップ検出（完全開放ドア）
        # =========================
        for i in range(len(segments) - 1):

            a = segments[i]
            b = segments[i + 1]

            if a.length < self.wall_min_len:
                continue

            gap_dist = float(np.linalg.norm(b.start - a.end))

            if not (self.door_width_min <= gap_dist <= self.door_width_max):
                continue

            angle_diff = self._angle_between(a.direction, b.direction)

            if angle_diff > self.collinear_angle_thresh * 2:
                continue

            center = (a.end + b.start) / 2.0

            yaw = float(np.arctan2(
                a.direction[1],
                a.direction[0]
            ))

            candidates.append(DoorCandidate(
                position=center,
                width=gap_dist,
                yaw=yaw,
                state="open",
                confidence=0.7
            ))

        return candidates
    
    def _merge_candidates(self, candidates):
        merged = []

        for c in candidates:
            found = False

            for m in merged:
                if np.linalg.norm(c.position - m.position) < self.pos_tolerance:
                    found = True
                    break

            if not found:
                merged.append(c)

        return merged

    def _check_floor_continuity(self, door: DoorCandidate) -> bool:
        if self._raw_points is None:
            return True
        pos = door.position
        near_floor = self._raw_points[
            np.abs(self._raw_points[:, 2] - self.floor_check_z) < 0.05
        ]
        if len(near_floor) == 0:
            return True
        dists = np.linalg.norm(near_floor[:, :2] - pos, axis=1)
        return bool(np.any(dists < self.floor_check_tol))

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

    def _cluster_segments(self, points: np.ndarray) -> List[Segment]:

        if len(points) < self.min_seg_pts:
            return []

        clustering = DBSCAN(
            eps=self.gap_thresh,
            min_samples=self.min_seg_pts
        ).fit(points)

        labels = clustering.labels_

        segments = []

        for label in np.unique(labels):

            if label == -1:
                continue

            cluster = points[labels == label]

            seg = self._fit_segment(cluster)

            if seg is not None:
                segments.append(seg)

        return segments
    
    def _sort_segments(self, segments):

        if len(segments) <= 1:
            return segments

        return sorted(
            segments,
            key=lambda s: (
                (s.start[0] + s.end[0]) / 2.0,
                (s.start[1] + s.end[1]) / 2.0
            )
        )
    
    def _merge_collinear_segments(self, segments):

        if len(segments) < 2:
            return segments

        merged = []

        current = segments[0]

        for nxt in segments[1:]:

            angle = self._angle_between(
                current.direction,
                nxt.direction
            )

            gap = np.linalg.norm(
                nxt.start - current.end
            )

            if (
                angle < np.deg2rad(8)
                and gap < 0.20
            ):

                pts = np.vstack([
                    current.points,
                    nxt.points
                ])

                current = self._fit_segment(pts)

            else:

                merged.append(current)

                current = nxt

        merged.append(current)

        return merged

    def _fit_segment(self, pts: np.ndarray) -> Optional[Segment]:
        if len(pts) < self.min_seg_pts:
            return None
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        if cov.ndim < 2:
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        normal = np.array([-direction[1], direction[0]])
        proj = (pts - center) @ direction
        length = float(proj.max() - proj.min())
        residuals = np.abs((pts - center) @ normal)
        if residuals.std() > self.fit_thresh:
            return None
        return Segment(
            pts,
            center + proj.min() * direction,
            center + proj.max() * direction,
            direction, normal, length, float(residuals.std())
        )

    def _angle_between(self, d1, d2):
        return float(np.arccos(np.clip(np.abs(d1 @ d2), 0.0, 1.0)))

    def _update_stable_doors(self, candidates: List[DoorCandidate]):
        matched_ids = set()
        for c in candidates:
            best_id = None
            best_dist = float('inf')
            for did, stable in self.stable_doors.items():
                dist = float(np.linalg.norm(c.position - stable.position))
                if dist < self.pos_tolerance and dist < best_dist:
                    best_dist = dist
                    best_id = did
            if best_id is not None:
                self.stable_doors[best_id].position = c.position
                self.stable_doors[best_id].width = c.width
                self.stable_doors[best_id].yaw = c.yaw
                self.stable_doors[best_id].state = c.state   
                self.stable_doors[best_id].update(c.state)
                matched_ids.add(best_id)
            else:
                sd = StableDoor(position=c.position, width=c.width, yaw=c.yaw, state=c.state)
                sd.update(c.state)
                self.stable_doors[self.next_door_id] = sd
                matched_ids.add(self.next_door_id)
                self.next_door_id += 1

        for did in list(self.stable_doors.keys()):
            if did not in matched_ids:
                self.stable_doors[did].history.append('unknown')
                if self.stable_doors[did].history.count('unknown') >= 7:
                    del self.stable_doors[did]

    def _publish_markers(self, stable: list):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        delete_all = Marker()
        delete_all.header.frame_id = 'lidar_link'
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for did, door in stable:
            state = door.stable_state()
            half_w = door.width / 2.0
            dx = np.cos(door.yaw) * half_w
            dy = np.sin(door.yaw) * half_w

            m = Marker()
            m.header.frame_id = 'lidar_link'
            m.header.stamp = now
            m.ns = 'door_frame'
            m.id = did
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
            p1 = Point(); p1.x, p1.y = float(door.position[0] - dx), float(door.position[1] - dy)
            p2 = Point(); p2.x, p2.y = float(door.position[0] + dx), float(door.position[1] + dy)
            m.points = [p1, p2]
            marker_array.markers.append(m)

            t = Marker()
            t.header.frame_id = 'lidar_link'
            t.header.stamp = now
            t.ns = 'door_label'
            t.id = did
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(door.position[0])
            t.pose.position.y = float(door.position[1])
            t.pose.position.z = 0.5
            t.scale.z = 0.2
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"Door{did}: {state.upper()}\nw={door.width:.2f}m"
            marker_array.markers.append(t)

        self.pub_markers.publish(marker_array)

    def _publish_info(self, stable: list):
        if not stable:
            return
        lines = [
            f"Door{did}: {d.stable_state()} pos=({d.position[0]:.2f},{d.position[1]:.2f}) width={d.width:.2f}m"
            for did, d in stable
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
