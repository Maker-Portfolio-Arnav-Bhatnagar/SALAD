# banana_detector.py:
# Takes Realsense color, aligned depth & camera-info input and identifies one sideways banana
# Returns the same midpoint, 4-corner box, surface height & long-axis orientation as carrot_detector.py
# Keeps the newest valid result available for SALAD_V1 and publishes it for debugging

from __future__ import annotations

import rclpy

from carrot_detector import CarrotDetection, CarrotDetector


# Both detectors return identical geometry, so keep a banana-specific public type name
BananaDetection = CarrotDetection


class BananaDetector(CarrotDetector):

    def __init__(self, show_debug: bool = True):
        # HSV range for a ripe yellow banana - tune these values if the room lighting changes
        super().__init__(
            show_debug=show_debug,
            object_name='banana',
            lower_hsv=(18, 70, 70),
            upper_hsv=(38, 255, 255),
        )

        # Bananas can occupy a slightly larger or less saturated region than carrots
        self.minimum_area = 1000.0


def main(args=None):
    rclpy.init(args=args)
    node = BananaDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
