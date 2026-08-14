# test_banana_detector.py:
# Runs banana_detector.py using the live Realsense feed and displays the detection windows
# Prints the banana midpoint, 4 bounding-box corners, orientation, depth & height to the terminal

import math

import rclpy

from banana_detector import BananaDetector


class BananaDetectorTest(BananaDetector):

    def __init__(self):
        super().__init__(show_debug=True)
        self._last_printed_stamp = -1.0

        # Check for a new result separately so the detector code remains unchanged
        self._print_timer = self.create_timer(0.10, self.print_latest_detection)
        self.get_logger().info(
            "Live banana detector test started - press Ctrl+C in this terminal to stop"
        )

    def print_latest_detection(self) -> None:
        """Print each new banana detection once."""
        detection = self.latest_detection
        if detection is None or detection.stamp_seconds == self._last_printed_stamp:
            return

        self._last_printed_stamp = detection.stamp_seconds
        x, y, z = detection.midpoint_camera
        angle_degrees = math.degrees(detection.angle_camera)

        self.get_logger().info(
            "\n"
            f"Banana midpoint pixel: {detection.midpoint_pixel}\n"
            f"Banana midpoint camera coords (m): "
            f"x={x:.4f}, y={y:.4f}, z={z:.4f}\n"
            f"Bounding-box corners pixels: {detection.corners_pixel}\n"
            f"Bounding-box corners camera coords (m): {detection.corners_camera}\n"
            f"Banana orientation: {detection.angle_camera:.4f} rad "
            f"({angle_degrees:.2f} deg)\n"
            f"Estimated height above surface: {detection.surface_height:.4f} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BananaDetectorTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Banana detector test stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
