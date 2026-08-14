# carrot_detector.py:
# Takes Realsense color, aligned depth & camera-info input and identifies one sideways carrot
# Returns a tight 4-corner box, carrot midpoint, surface height & long-axis orientation
# Keeps the newest valid result available for SALAD_V1 and publishes it as JSON for debugging

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image as ROSImage
from std_msgs.msg import String


@dataclass(frozen=True)
class CarrotDetection:
    """All geometry produced for one detected carrot."""

    midpoint_pixel: tuple[int, int]
    midpoint_camera: tuple[float, float, float]
    corners_pixel: tuple[tuple[int, int], ...]
    corners_camera: tuple[tuple[float, float, float], ...]
    angle_camera: float
    surface_height: float
    contour_area: float
    stamp_seconds: float


class CarrotDetector(Node):

    def __init__(self, show_debug: bool = True, *, object_name: str = 'carrot',
                 lower_hsv=(3, 90, 60), upper_hsv=(28, 255, 255)):
        super().__init__(f'{object_name}_detector')

        # Initialize CV Bridge & newest sensor values
        self._bridge = CvBridge()
        self._depth_image: Optional[np.ndarray] = None
        self._depth_stamp_seconds = 0.0
        self._camera_matrix: Optional[np.ndarray] = None
        self.latest_detection: Optional[CarrotDetection] = None
        self.show_debug = show_debug
        self.object_name = object_name

        # Detection parameters - HSV values may need small adjustments for the room lighting
        self.lower_orange = np.array(lower_hsv, dtype=np.uint8)
        self.upper_orange = np.array(upper_hsv, dtype=np.uint8)
        self.minimum_area = 1000.0
        self.maximum_depth_age = 0.20

        # Create subscriptions
        self.image_sub = self.create_subscription(
            ROSImage,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            ROSImage,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10,
        )
        self.detection_pub = self.create_publisher(
            String, f'/{self.object_name}/detection', 10
        )

        self.get_logger().info(f"{self.object_name.capitalize()} detector started")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Store camera intrinsics used to turn a depth pixel into metres."""
        self._camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def depth_callback(self, msg: ROSImage) -> None:
        """Store latest aligned depth image in metres."""
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = np.asarray(depth, dtype=np.float32)

        # Realsense 16-bit aligned depth is millimetres; 32-bit depth is normally metres
        if msg.encoding in ('16UC1', 'mono16') or np.nanmedian(depth) > 20.0:
            depth *= 0.001

        self._depth_image = depth
        self._depth_stamp_seconds = self._stamp_seconds(msg)

    @staticmethod
    def _stamp_seconds(msg: ROSImage) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _find_carrot_contour(image: np.ndarray, lower: np.ndarray,
                             upper: np.ndarray, minimum_area: float):
        """Create a clean orange mask & return its largest carrot-sized contour."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        contour = max(contours, key=cv2.contourArea) if contours else None
        return contour, mask

    @staticmethod
    def _box_and_angle(contour) -> tuple[np.ndarray, float]:
        """Return ordered rectangle corners & the carrot long-axis angle in image coordinates."""
        rectangle = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rectangle).astype(np.float32)

        # Pick the longest rectangle edge, avoiding OpenCV's version-dependent angle rules
        edges = np.roll(corners, -1, axis=0) - corners
        long_edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        angle = math.atan2(float(long_edge[1]), float(long_edge[0]))

        # A carrot axis has no forward/backward direction, so keep angle in [-pi/2, pi/2)
        if angle >= math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        return corners, angle

    def _valid_depth(self, x: int, y: int, radius: int = 3) -> Optional[float]:
        """Use a local median so a single missing depth pixel does not discard a detection."""
        if self._depth_image is None:
            return None
        height, width = self._depth_image.shape[:2]
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = self._depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        return float(np.median(valid)) if valid.size else None

    def _deproject(self, pixel: tuple[int, int], depth: float) -> tuple[float, float, float]:
        """Deproject one aligned color pixel into the Realsense optical frame."""
        if self._camera_matrix is None:
            raise RuntimeError("Camera intrinsics have not arrived")
        x, y = pixel
        fx, fy = self._camera_matrix[0, 0], self._camera_matrix[1, 1]
        cx, cy = self._camera_matrix[0, 2], self._camera_matrix[1, 2]
        return (
            float((x - cx) * depth / fx),
            float((y - cy) * depth / fy),
            float(depth),
        )

    def _estimate_surface_height(self, contour, carrot_depth: float) -> float:
        """Estimate carrot height using nearby table depth minus carrot surface depth."""
        contour_mask = np.zeros(self._depth_image.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        outer = cv2.dilate(contour_mask, np.ones((31, 31), np.uint8))
        ring = (outer > 0) & (contour_mask == 0)
        values = self._depth_image[ring]
        values = values[np.isfinite(values) & (values > 0.05) & (values < 3.0)]
        return max(0.0, float(np.median(values) - carrot_depth)) if values.size else 0.0

    def image_callback(self, msg: ROSImage) -> None:
        """Identify the carrot & store its newest complete 3D detection."""
        if self._depth_image is None or self._camera_matrix is None:
            return

        color_stamp = self._stamp_seconds(msg)
        if abs(color_stamp - self._depth_stamp_seconds) > self.maximum_depth_age:
            return

        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        display = image.copy()
        contour, mask = self._find_carrot_contour(
            image, self.lower_orange, self.upper_orange, self.minimum_area
        )

        if contour is not None:
            corners_float, angle = self._box_and_angle(contour)
            moments = cv2.moments(contour)
            if moments['m00'] > 0.0:
                midpoint = (
                    int(round(moments['m10'] / moments['m00'])),
                    int(round(moments['m01'] / moments['m00'])),
                )
                midpoint_depth = self._valid_depth(*midpoint)
                corners = tuple((int(round(x)), int(round(y))) for x, y in corners_float)
                corner_depths = [self._valid_depth(*corner) for corner in corners]

                if midpoint_depth is not None and all(depth is not None for depth in corner_depths):
                    midpoint_camera = self._deproject(midpoint, midpoint_depth)
                    corners_camera = tuple(
                        self._deproject(corner, depth)
                        for corner, depth in zip(corners, corner_depths)
                    )
                    detection = CarrotDetection(
                        midpoint_pixel=midpoint,
                        midpoint_camera=midpoint_camera,
                        corners_pixel=corners,
                        corners_camera=corners_camera,
                        angle_camera=angle,
                        surface_height=self._estimate_surface_height(contour, midpoint_depth),
                        contour_area=float(cv2.contourArea(contour)),
                        stamp_seconds=color_stamp,
                    )
                    self.latest_detection = detection
                    self.detection_pub.publish(String(data=json.dumps(asdict(detection))))

                    # Highlight all returned image coordinates with blue dots
                    cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                    for corner in corners:
                        cv2.circle(display, corner, 5, (255, 0, 0), -1)
                    cv2.circle(display, midpoint, 6, (255, 0, 0), -1)
                    axis_end = (
                        int(midpoint[0] + 70 * math.cos(angle)),
                        int(midpoint[1] + 70 * math.sin(angle)),
                    )
                    cv2.line(display, midpoint, axis_end, (255, 0, 0), 2)

        if self.show_debug:
            cv2.imshow(f'{self.object_name.capitalize()} Mask', mask)
            cv2.imshow(f'{self.object_name.capitalize()} Detection', display)
            cv2.waitKey(1)

    def wait_for_detection(self, executor, timeout: float = 15.0,
                           stable_frames: int = 5) -> CarrotDetection:
        """Wait for several mutually consistent detections before robot motion begins."""
        deadline = time.monotonic() + timeout
        samples: list[CarrotDetection] = []
        last_stamp = -1.0

        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            detection = self.latest_detection
            if detection is None or detection.stamp_seconds == last_stamp:
                continue
            last_stamp = detection.stamp_seconds
            samples.append(detection)
            samples = samples[-stable_frames:]

            if len(samples) == stable_frames:
                points = np.asarray([sample.midpoint_camera for sample in samples])
                angles = np.asarray([sample.angle_camera for sample in samples])

                # Compare doubled angles because +90 & -90 degrees are the same carrot axis
                mean_axis = 0.5 * math.atan2(
                    float(np.mean(np.sin(2.0 * angles))),
                    float(np.mean(np.cos(2.0 * angles))),
                )
                angle_errors = 0.5 * np.arctan2(
                    np.sin(2.0 * (angles - mean_axis)),
                    np.cos(2.0 * (angles - mean_axis)),
                )
                if (np.max(np.ptp(points, axis=0)) < 0.015
                        and np.max(np.abs(angle_errors)) < math.radians(8)):
                    return samples[-1]

        raise TimeoutError(
            f"No stable {self.object_name} detection was received before timeout"
        )

    def destroy_node(self):
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarrotDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
