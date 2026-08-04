#!/usr/bin/env python3
"""
aruco_detector1.py  —  ROS 2 ArUco Detection + Pose Transform Node
====================================================================

WHAT THIS NODE DOES
───────────────────
1. Subscribes to the live RGB camera stream and camera intrinsics.
2. Detects ArUco markers (DICT_4X4_50) in every frame using OpenCV.
3. Filters to ONLY process marker ID 7 — all other IDs are ignored.
4. Estimates the 6-DoF pose of marker ID 7 in the CAMERA optical frame.
5. Transforms that pose into the ROBOT BASE frame using:
       T_base_object = T_BASE_LINK @ T_LINK_OPTICAL @ T_optical_object
6. Publishes THREE outputs:
     /aruco/id7/pose          — PoseStamped        in robot base frame
     /transformed_pos         — PoseStamped        in robot base frame
     /transformed_pos_euler   — Float64MultiArray  [x, y, z, r, p, y] (radians)
7. Logs every transformed pose to CSV (appends on restart):
     ~/bimanual_ws/src/ds_control/scripts/aruco_positions.csv

SUBSCRIBED TOPICS
─────────────────
  /camera/camera/color/image_raw      sensor_msgs/Image
  /camera/camera/color/camera_info    sensor_msgs/CameraInfo

PUBLISHED TOPICS
────────────────
  /aruco/id7/pose              geometry_msgs/PoseStamped
  /transformed_pos             geometry_msgs/PoseStamped
  /transformed_pos_euler       std_msgs/Float64MultiArray

WHAT TO UPDATE WHEN HARDWARE CHANGES
──────────────────────────────────────
  • T_BASE_LINK  — re-calibrate and paste new 4×4 matrix if the
                   camera is moved relative to the robot base.
  • marker_size  — update if the physical ArUco marker size changes.
  • DICT_4X4_50  — update if a different ArUco dictionary is used.
"""

import os
import csv
import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image as ROSImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORM CHAIN
#  T_base_object = T_BASE_LINK @ T_LINK_OPTICAL @ T_optical_object
#
#  T_BASE_LINK    — calibrated camera-to-robot-base transform
#  T_LINK_OPTICAL — corrects axis mismatch between camera link and optical frame
# ══════════════════════════════════════════════════════════════════════════════

T_BASE_LINK = np.array([
    [ 0.0378, -0.9950, -0.0927, -0.4213],
    [-0.1580, -0.0976,  0.9826,  0.4308],
    [-0.9867, -0.0225, -0.1609,  1.2300],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
], dtype=np.float64)

T_LINK_OPTICAL = np.array([
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1],
], dtype=np.float64)

# Combined: camera optical frame → robot base frame
T_CAM_TO_ROBOT = T_BASE_LINK @ T_LINK_OPTICAL


def to_robot_frame(pos_cam, quat_cam):
    """
    Transforms position and orientation from camera optical frame
    to robot base frame using:
        T_base_object = T_BASE_LINK @ T_LINK_OPTICAL @ T_optical_object

    pos_cam  : [x, y, z]     in camera optical frame
    quat_cam : [x, y, z, w]  in camera optical frame
    Returns    pos_robot [x, y, z], quat_robot [x, y, z, w]
    """
    # Position
    p_cam   = np.array([*pos_cam, 1.0])
    p_robot = (T_CAM_TO_ROBOT @ p_cam)[:3]

    # Orientation
    R_cam      = R.from_quat(quat_cam).as_matrix()
    R_robot    = T_CAM_TO_ROBOT[:3, :3] @ R_cam
    quat_robot = R.from_matrix(R_robot).as_quat()   # [x, y, z, w]

    return p_robot.tolist(), quat_robot.tolist()


# ── Target marker ID — only this ID is processed, all others ignored ──────────
TARGET_ID = 7


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self._bridge = CvBridge()

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

        # Camera intrinsics — filled from /camera_info topic
        self.camera_matrix = None
        self.dist_coeffs   = None

        # Marker physical size in metres
        self.marker_size = 0.077

        # ── Fixed publishers for ID 7 (created once at startup) ───────────────
        self._pub_aruco = self.create_publisher(
            PoseStamped, '/aruco/id7/pose', 10
        )
        self._pub_pose = self.create_publisher(
            PoseStamped, '/transformed_pos', 10
        )
        self._pub_euler = self.create_publisher(
            Float64MultiArray, '/transformed_pos_euler', 10
        )

        # ── CSV logging setup ──────────────────────────────────────────────────
        csv_path = os.path.expanduser(
            "~/bimanual_ws/src/ds_control/scripts/aruco_positions.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
        self._csv_file   = open(csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(["time_sec", "marker_id",
                                       "x", "y", "z",
                                       "roll", "pitch", "yaw"])
            self._csv_file.flush()
        self.get_logger().info(f"CSV logging to: {csv_path}")

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(
            ROSImage,
            '/camera/camera/color/image_raw',
            self._image_callback,
            10
        )
        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self._camera_info_callback,
            10
        )

        self.get_logger().info(
            f"ArUco detector started — tracking ID {TARGET_ID} only | "
            f"publishing /transformed_pos and /transformed_pos_euler (radians)"
        )

    # --------------------------------------------------------------------------
    def _camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received.")

    # --------------------------------------------------------------------------
    def _image_callback(self, msg: ROSImage):
        if self.camera_matrix is None:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )

        if ids is None:
            cv2.imshow("ArUco Detection", frame)
            cv2.waitKey(1)
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
        )

        now_msg = self.get_clock().now().to_msg()
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        for i, marker_id in enumerate(ids.flatten()):

            # ── FILTER: skip every marker that is not ID 7 ────────────────────
            # This prevents false positives (e.g. ID 37) from being processed
            if marker_id != TARGET_ID:
                continue

            # Draw axis on the frame for the detected marker
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs,
                rvecs[i], tvecs[i], 0.03
            )

            # ── Position and orientation in camera optical frame ───────────────
            pos_cam = [float(tvecs[i][0][0]),
                       float(tvecs[i][0][1]),
                       float(tvecs[i][0][2])]

            rot_mat, _ = cv2.Rodrigues(rvecs[i])
            quat_cam   = R.from_matrix(rot_mat).as_quat()   # [x, y, z, w]

            # ── Transform to robot base frame ──────────────────────────────────
            pos_robot, quat_robot = to_robot_frame(pos_cam, quat_cam.tolist())

            # ── Build PoseStamped message (shared by pub 1 and pub 2) ──────────
            pose_msg = PoseStamped()
            pose_msg.header.stamp    = now_msg
            pose_msg.header.frame_id = "base_link"
            pose_msg.pose.position.x    = pos_robot[0]
            pose_msg.pose.position.y    = pos_robot[1]
            pose_msg.pose.position.z    = pos_robot[2]
            pose_msg.pose.orientation.x = quat_robot[0]
            pose_msg.pose.orientation.y = quat_robot[1]
            pose_msg.pose.orientation.z = quat_robot[2]
            pose_msg.pose.orientation.w = quat_robot[3]

            # ── Publisher 1: /aruco/id7/pose ───────────────────────────────────
            self._pub_aruco.publish(pose_msg)

            # ── Publisher 2: /transformed_pos ──────────────────────────────────
            self._pub_pose.publish(pose_msg)

            # ── Euler angles in radians ────────────────────────────────────────
            r_obj              = R.from_quat(quat_robot)
            roll, pitch, yaw   = r_obj.as_euler('xyz', degrees=False)

            # ── Publisher 3: /transformed_pos_euler ────────────────────────────
            arr = Float64MultiArray()
            arr.layout.dim = [
                MultiArrayDimension(label="fields", size=6, stride=6)
            ]
            arr.data = [
                pos_robot[0], pos_robot[1], pos_robot[2],
                roll, pitch, yaw
            ]
            self._pub_euler.publish(arr)

            # ── CSV logging ────────────────────────────────────────────────────
            try:
                self._csv_writer.writerow([
                    now_sec, marker_id,
                    pos_robot[0], pos_robot[1], pos_robot[2],
                    roll, pitch, yaw
                ])
                self._csv_file.flush()
            except Exception as e:
                self.get_logger().warn(f"CSV write failed: {e}")

            # ── Terminal log ───────────────────────────────────────────────────
            self.get_logger().info(
                f"ID {marker_id} | "
                f"cam:   x={pos_cam[0]:.3f}  y={pos_cam[1]:.3f}  z={pos_cam[2]:.3f} | "
                f"robot: x={pos_robot[0]:.3f}  y={pos_robot[1]:.3f}  z={pos_robot[2]:.3f} | "
                f"rpy(rad): r={roll:.3f}  p={pitch:.3f}  y={yaw:.3f}"
            )

            # ── Overlay on frame ───────────────────────────────────────────────
            cv2.putText(
                frame,
                f"ID{marker_id} [robot]: x={pos_robot[0]:.2f} "
                f"y={pos_robot[1]:.2f} z={pos_robot[2]:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
            )

        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)

    # --------------------------------------------------------------------------
    def destroy_node(self):
        """Clean up CSV file handle on shutdown."""
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        self.get_logger().info("CSV closed. Node shutting down.")
        super().destroy_node()


# ------------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()