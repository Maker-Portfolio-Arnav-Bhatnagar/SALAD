import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

class banana_detect(Node):

    def __init__(self):

        super().__init__('BANANA_DETECTOR')

        # Initialize CV Bridge
        self._bridge = CvBridge()

        # Create subscriptions
        self.image_sub = self.create_subscription(
                    ROSImage,
                    '/camera/camera/color/image_raw',
                    self.image_callback,
                    10
                    )
        self.depth_sub = self.create_subscription(
                    ROSImage,
                    '/camera/camera/aligned_depth_to_color/image_raw',
                    self.depth_callback,
                    10
                    )
        
    def image_callback(self, msg)-> None:
        """Store latest color image, and identifies the coords of the entire mask"""

        self.get_logger().info("Received image")
        self.color_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        image = self.color_image.copy()

        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV range for yellow/orange banana. These numbers will likely need tuning.
        lower = (10, 80, 80)
        upper = (40, 255, 255)

        mask = cv2.inRange(hsv_image, lower, upper)

        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            # Largest detected object
            contour = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(contour)

            # Ignore tiny blobs
            if area > 1000:

                # Draw contour
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                # Compute centroid
                M = cv2.moments(contour)

                if M["m00"] != 0:

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw midpoint
                    cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)

                    cv2.putText(
                        image,
                        "({}, {})".format(cx, cy),
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    print("Center:", cx, cy)

        cv2.imshow("Mask", mask)
        cv2.imshow("Camera", image)
        cv2.waitKey(1)

    def depth_callback(self, msg)-> None:
        """Store latest depth image."""
        self._depth_image = self._bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3


def main(args=None):
    rclpy.init(args=args)

    node = banana_detect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


