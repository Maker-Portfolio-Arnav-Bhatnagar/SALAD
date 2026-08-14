# live_feed_plain.py:
# Displays the Realsense color camera feed without performing any detection

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class LiveFeed(Node):

    def __init__(self):
        super().__init__('live_feed_plain')

        # Converts ROS image messages into images that OpenCV can display
        self.bridge = CvBridge()

        # Subscribe to the Realsense color camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10,
        )

    def image_callback(self, msg):
        # Convert the ROS image into a normal BGR image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display the newest camera frame
        cv2.imshow('Live Camera Feed', image)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveFeed()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
