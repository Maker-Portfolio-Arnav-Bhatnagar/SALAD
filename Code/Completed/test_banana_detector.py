# test_banana_detector.py:
# Runs banana_detector.py using the live Realsense feed and displays the detection windows
# Prints the first banana result that remains consistent for 3 camera frames

import inspect
import math

import rclpy
from rclpy.executors import SingleThreadedExecutor

from banana_detector import BananaDetector
from coordinate_transformer import (
    transform_object_angle,
    transform_point,
    transform_points,
)


def print_detection(node, detection):
    """Print one stable banana detection to the terminal."""
    # Convert the detector's camera-frame result into the Franka base frame
    midpoint_franka = transform_point(detection.midpoint_camera)
    corners_franka = transform_points(detection.corners_camera)
    angle_franka = transform_object_angle(detection.angle_camera)

    x, y, z = midpoint_franka
    angle_degrees = math.degrees(angle_franka)

    # Convert NumPy arrays into normal lists so they print clearly
    corners_franka = corners_franka.tolist()

    node.get_logger().info(
        "\n"
        f"Stable banana midpoint pixel: {detection.midpoint_pixel}\n"
        f"Stable banana midpoint Franka coords (m): "
        f"x={x:.4f}, y={y:.4f}, z={z:.4f}\n"
        f"Bounding-box corners pixels: {detection.corners_pixel}\n"
        f"Bounding-box corners Franka coords (m): {corners_franka}\n"
        f"Banana orientation in Franka frame: {angle_franka:.4f} rad "
        f"({angle_degrees:.2f} deg)\n"
        f"Estimated height above surface: {detection.surface_height:.4f} m"
    )


def main(args=None):
    rclpy.init(args=args)
    node = BananaDetector(show_debug=True)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.get_logger().info(
        f"Loaded detector code from: {inspect.getfile(BananaDetector)}"
    )
    node.get_logger().info("Waiting for 3 consistent banana detections")

    try:
        # Select the first result that stays consistent for 3 camera frames
        detection = node.wait_for_detection(
            executor,
            timeout=20.0,
            stable_frames=3,
        )
        print_detection(node, detection)

        # Keep updating the live feed without printing fluctuating values
        node.get_logger().info("Live feed continuing - press Ctrl+C to stop")
        executor.spin()
    except TimeoutError as error:
        node.get_logger().warn(str(error))
    except KeyboardInterrupt:
        node.get_logger().info("Banana detector test stopped")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
