#!/usr/bin/env python3

# Mopel Kitele and Logan Purkiss

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist


class RotateRobot(Node):

    def __init__(self):
        super().__init__('rotate_robot')

        # Camera width is 320 px, so center is x = 160
        self.image_center_x = 160.0

        # Small center tolerance so the robot does not jitter
        self.deadband = 20.0

        # Rotation speed
        self.angular_speed = 0.7

        self.centroid_subscriber = self.create_subscription(
            Point,
            '/tracking/centroid',
            self.centroid_callback,
            10
        )

        self.velocity_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info("RotateRobot node has been started.")

    def centroid_callback(self, msg):
        error = msg.x - self.image_center_x

        cmd = Twist()

        # Lab requires rotation only
        cmd.linear.x = 0.0

        if error > self.deadband:
            # Target is on the right side of the image
            cmd.angular.z = -self.angular_speed

        elif error < -self.deadband:
            # Target is on the left side of the image
            cmd.angular.z = self.angular_speed

        else:
            # Target is centered
            cmd.angular.z = 0.0

        self.velocity_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = RotateRobot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.velocity_publisher.publish(stop_cmd)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
