# banana_detector.py:
# Takes Realsense color, aligned depth & camera-info input and identifies one sideways banana
# Returns a tight 4-corner box, banana midpoint, surface height & long-axis orientation
# Keeps the newest valid result available for SALAD_V1 and publishes it as JSON for debugging
# All distances returned by this program are in metres and all angles are in radians

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


# These values select yellow pixels in an HSV image
# Change these first if the banana is not detected under different lighting
LOWER_YELLOW_HSV = np.array([18, 70, 70], dtype=np.uint8)
UPPER_YELLOW_HSV = np.array([38, 255, 255], dtype=np.uint8)

MINIMUM_BANANA_AREA = 300.0 # Ignore tiny yellow dots, but allow a distant banana
MAXIMUM_DEPTH_AGE = 0.20 # Color and depth frames must be captured within this many seconds of each other

@dataclass(frozen=True)
class BananaDetection:
    # Pixel values are useful for drawing on the camera image
    # Camera values are 3D [x, y, z] positions measured in metres

    midpoint_pixel: tuple[int, int]
    midpoint_camera: tuple[float, float, float]
    corners_pixel: tuple[tuple[int, int], ...]
    corners_camera: tuple[tuple[float, float, float], ...]
    angle_camera: float
    surface_height: float
    contour_area: float
    stamp_seconds: float


class BananaDetector(Node):

    def __init__(self, show_debug: bool = True):
        super().__init__('banana_detector')

        # CV Bridge converts ROS image messages into OpenCV images
        self._bridge = CvBridge()

        # These are filled when messages arrive from the Realsense camera
        self._depth_image: Optional[np.ndarray] = None
        self._depth_stamp_seconds = 0.0
        self._camera_matrix: Optional[np.ndarray] = None

        # SALAD_V1 reads this variable after a banana is found
        self.latest_detection: Optional[BananaDetection] = None
        self.show_debug = show_debug

        # Subscribe to the color image, aligned depth image and camera calibration
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
        
        # The JSON topic makes it easy to inspect results with ros2 topic echo
        self.detection_pub = self.create_publisher(String, '/banana/detection', 10)

        self.get_logger().info("Banana detector started")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Store the calibration values used to convert a pixel into a 3D point."""
        self._camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def depth_callback(self, msg: ROSImage) -> None:
        """Store latest aligned depth image in metres."""
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = np.asarray(depth, dtype=np.float32)

        # Realsense 16-bit depth is in millimetres
        # The rest of this project uses metres, so divide these values by 1000
        if msg.encoding in ('16UC1', 'mono16') or np.nanmedian(depth) > 20.0:
            depth *= 0.001

        self._depth_image = depth
        self._depth_stamp_seconds = self._stamp_seconds(msg)

    @staticmethod
    def _stamp_seconds(msg: ROSImage) -> float:
        """Convert a ROS timestamp into one floating-point seconds value."""
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _find_banana_contour(image: np.ndarray):
        """Create a clean yellow mask and return the largest banana-shaped region."""
        # HSV separates color from brightness better than the camera's BGR format
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, LOWER_YELLOW_HSV, UPPER_YELLOW_HSV)

        # Opening removes isolated dots; closing fills small holes in the banana
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, mask

        # Assume the largest yellow object is the banana
        banana_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(banana_contour) < MINIMUM_BANANA_AREA:
            return None, mask

        return banana_contour, mask

    @staticmethod
    def _box_and_angle(contour) -> tuple[np.ndarray, float]:
        """Return ordered rectangle corners & the banana long-axis angle in image coordinates."""
        rectangle = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rectangle).astype(np.float32)

        # Work out which side of the box is longest
        edges = np.roll(corners, -1, axis=0) - corners
        long_edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        angle = math.atan2(float(long_edge[1]), float(long_edge[0]))

        # A banana axis has no forward/backward direction, so keep angle in [-pi/2, pi/2)
        if angle >= math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        return corners, angle

    def _valid_depth(self, x: int, y: int, radius: int = 3) -> Optional[float]:
        """Use a local median so a single missing depth pixel does not discard a detection."""
        if self._depth_image is None:
            return None

        # Read a small square around the requested pixel
        height, width = self._depth_image.shape[:2]
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = self._depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        if valid.size == 0:
            return None

        # The median is not badly affected by one incorrect depth pixel
        return float(np.median(valid))

    def _deproject(self, pixel: tuple[int, int], depth: float) -> tuple[float, float, float]:
        """Deproject one aligned color pixel into the Realsense optical frame."""
        if self._camera_matrix is None:
            raise RuntimeError("Camera intrinsics have not arrived")
        pixel_x, pixel_y = pixel
        fx, fy = self._camera_matrix[0, 0], self._camera_matrix[1, 1]
        cx, cy = self._camera_matrix[0, 2], self._camera_matrix[1, 2]

        # Standard pinhole-camera equations
        camera_x = (pixel_x - cx) * depth / fx
        camera_y = (pixel_y - cy) * depth / fy
        return (
            float(camera_x),
            float(camera_y),
            float(depth),
        )

    def _estimate_surface_height(self, contour, banana_depth: float) -> float:
        """Estimate banana height using nearby table depth minus banana surface depth."""
        # Make a filled mask of the banana
        contour_mask = np.zeros(self._depth_image.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

        # The ring just outside that mask should contain the table surface
        outer = cv2.dilate(contour_mask, np.ones((31, 31), np.uint8))
        ring = (outer > 0) & (contour_mask == 0)
        values = self._depth_image[ring]
        values = values[np.isfinite(values) & (values > 0.05) & (values < 3.0)]
        if values.size == 0:
            return 0.0

        table_depth = float(np.median(values))
        return max(0.0, table_depth - banana_depth)

    def image_callback(self, msg: ROSImage) -> None:
        """Identify the banana & store its newest complete 3D detection."""
        # A 3D result cannot be calculated until depth and calibration arrive
        if self._depth_image is None or self._camera_matrix is None:
            return

        color_stamp = self._stamp_seconds(msg)
        if abs(color_stamp - self._depth_stamp_seconds) > MAXIMUM_DEPTH_AGE:
            return

        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        display = image.copy()
        contour, mask = self._find_banana_contour(image)

        if contour is not None:
            contour_area = cv2.contourArea(contour)
            cv2.putText(
                display,
                f'BANANA FOUND - area: {contour_area:.0f}',
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Draw these as soon as a contour is found - depth is not needed
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 3)
            corners_float, angle = self._box_and_angle(contour)
            corners = tuple((int(round(x)), int(round(y))) for x, y in corners_float)
            cv2.polylines(
                display,
                [np.asarray(corners, dtype=np.int32)],
                True,
                (255, 0, 0),
                2,
            )
            for corner in corners:
                cv2.circle(display, corner, 7, (255, 0, 0), -1)

            moments = cv2.moments(contour)
            if moments['m00'] > 0.0:
                # Image midpoint calculated from the yellow region's moments
                midpoint = (
                    int(round(moments['m10'] / moments['m00'])),
                    int(round(moments['m01'] / moments['m00'])),
                )
                midpoint_depth = self._valid_depth(*midpoint)
                corner_depths = [self._valid_depth(*corner) for corner in corners]

                # Draw the banana midpoint
                cv2.circle(display, midpoint, 6, (255, 0, 0), -1)

                # Draw a blue line showing the banana's detected orientation
                axis_end = (
                    int(midpoint[0] + 70 * math.cos(angle)),
                    int(midpoint[1] + 70 * math.sin(angle)),
                )
                cv2.line(display, midpoint, axis_end, (255, 0, 0), 2)

                # Only publish after all five returned pixels have valid depth
                if midpoint_depth is not None and all(depth is not None for depth in corner_depths):
                    midpoint_camera = self._deproject(midpoint, midpoint_depth)
                    corners_camera = tuple(
                        self._deproject(corner, depth)
                        for corner, depth in zip(corners, corner_depths)
                    )
                    detection = BananaDetection(
                        midpoint_pixel=midpoint,
                        midpoint_camera=midpoint_camera,
                        corners_pixel=corners,
                        corners_camera=corners_camera,
                        angle_camera=angle,
                        surface_height=self._estimate_surface_height(contour, midpoint_depth),
                        contour_area=float(contour_area),
                        stamp_seconds=color_stamp,
                    )
                    self.latest_detection = detection
                    self.detection_pub.publish(String(data=json.dumps(asdict(detection))))
        else:
            cv2.putText(
                display,
                f'NO BANANA CONTOUR - minimum area: {MINIMUM_BANANA_AREA:.0f}',
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if self.show_debug:
            cv2.imshow('Banana Mask', mask)
            cv2.imshow('Banana Detection', display)
            cv2.waitKey(1)

    def wait_for_detection(self, executor, timeout: float = 15.0,
                           stable_frames: int = 5) -> BananaDetection:
        """Wait for several mutually consistent detections before robot motion begins."""
        deadline = time.monotonic() + timeout
        samples: list[BananaDetection] = []
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

                # Compare doubled angles because +90 & -90 degrees are the same banana axis
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

        raise TimeoutError("No stable banana detection was received before timeout")

    def destroy_node(self):
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


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


# FLOWCHART:
#
# Realsense sends color image, depth image & camera information
#                         |
#                         v
# Convert color image from BGR to HSV
#                         |
#                         v
# Keep yellow pixels & remove small amounts of noise
#                         |
#                         v
# Find the largest yellow contour
#                         |
#                  Banana found?
#                    /          \
#                  No            Yes
#                  |              |
#                  v              v
#          Display live feed   Find midpoint, tight box & orientation
#                                 |
#                                 v
#                       Read depth at returned pixels
#                                 |
#                                 v
#                     Convert pixels into 3D camera coords
#                                 |
#                                 v
#                  Save & publish the BananaDetection result
#                                 |
#                                 v
#                Draw contour, blue points & direction on feed
