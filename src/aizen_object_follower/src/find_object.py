#!/usr/bin/env python3

# Logan Purkiss and Mopel Kitele
import numpy as np
import cv2
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from rclpy.qos import (
QoSProfile,
QoSDurabilityPolicy,
QoSReliabilityPolicy,
QoSHistoryPolicy
)

class ObjectFinder(Node):

    def __init__(self):
        super().__init__('object_finder')

        self._bridge = CvBridge()

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
    )


        # Create a subscriber for the image topic
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            image_qos
        )
        self.hsv_subscriber = self.create_subscription(
            Float32MultiArray,
            '/tracking/target_hsv',
            self.hsv_callback,
            10
        )

        # Create a publisher for the HSV values
        self.centroid_publisher = self.create_publisher(
            Point,
            '/tracking/centroid',
            10
        )
        self.image_publisher = self.create_publisher(
            CompressedImage,
            '/tracking/processed_image',
            10
        )

        self.get_logger().info("ObjectFinder node has been started.")

    def hsv_callback(self, msg):
        # Store the target HSV values from the message
        if len(msg.data) != 6:
            self.get_logger().error("Received target HSV message does not contain 6 values.")
            return
        self.target_hsv = np.array(msg.data, dtype=np.float32)
        self.get_logger().info(f"Received target HSV values: {self.target_hsv[0:3]} and HSV threshold: {self.target_hsv[3:6]}")

    def image_callback(self, msg):
        # Convert the compressed image message to a numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        if not hasattr(self, 'target_hsv'):
            return

        found, cx, cy, contour = self.find_target(frame)

        if found:
            centroid_msg = Point()
            centroid_msg.x = float(cx)
            centroid_msg.y = float(cy)
            centroid_msg.z = 0.0

            # Draw a bounding box around the detected object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self.centroid_publisher.publish(centroid_msg)
            self.publish_compressed_image(frame)

    def find_target(self, frame):
        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_hsv = self.target_hsv[0:3]
        thresholds = self.target_hsv[3:6]

        # Define the lower and upper bounds for thresholding
        lower_bound = np.clip(target_hsv - thresholds, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        upper_bound = np.clip(target_hsv + thresholds, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

        # Threshold the HSV image to get only the colors in the range
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Run opening and closing to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = [
                contour for contour in contours
                if cv2.contourArea(contour) > 200
            ]

            if valid_contours:
                target_contour = max(valid_contours, key=cv2.contourArea)

                center = self.get_contour_center(target_contour)

                if center is not None:
                    cX, cY = center
                    return True, cX, cY, target_contour

        return False, None, None, None

    # Get the center of any contour
    def get_contour_center(self, contour):
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return None

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return (cX, cY)

    def publish_compressed_image(self, frame):
        # Convert the frame to a compressed image message
        out_msg = self._bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
        self.image_publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    object_finder = ObjectFinder()
    try:
        rclpy.spin(object_finder)
    except KeyboardInterrupt:
        pass
    finally:
        object_finder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
