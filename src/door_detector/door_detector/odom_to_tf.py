#!/usr/bin/env python3
"""odom_to_tf.py: /odom を購読して odom -> base_footprint の tf を発行する。
ロボットは world 座標 (1.0, 0.0) にスポーンするが、/odom は起動位置を原点とする
相対座標系のため、map(=world)座標に合わせて起動位置ぶんのオフセットを足す。"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter

# launchファイルの spawn_robot の -x, -y と一致させる
SPAWN_X = 1.0
SPAWN_Y = 0.0

class OdomToTf(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, '/odom', self.callback, 10)

    def callback(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        t.transform.translation.x = msg.pose.pose.position.x + SPAWN_X
        t.transform.translation.y = msg.pose.pose.position.y + SPAWN_Y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()