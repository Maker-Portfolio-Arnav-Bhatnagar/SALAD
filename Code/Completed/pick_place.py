# pick_place.py:
# Contains the Franka motion functions used by SALAD_V1 to pick up & place the banana
# Uses a safe approach/retreat waypoint and stops the sequence if any target times out

from __future__ import annotations

import math
import os
import sys
import time
from typing import Iterable

import numpy as np
from rclpy.executors import MultiThreadedExecutor


# Add the bimanual workspace source root when this file is run from scripts/Completed
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# Orientation measured from the working Franka reference program
DEFAULT_TOOL_QUAT = np.array([
    0.9208164128617827,
    -0.38895860442545854,
    -0.02843004791606157,
    -0.0002649966960190266,
], dtype=np.float64)

# Placeholder cutting-board location from the working Franka reference program
DEFAULT_PLACE_POS = [0.49262495730561356, -0.5186854477752654, 0.16798802875832192]
DEFAULT_PLACE_QUAT = [0.9251516457019573, -0.3795919664139947, 0.0012463626502968118, 0.0016787105351086956]
REST_POS = [0.3584641619510052, -0.036976077522332576, 0.49373634152165496]
REST_QUAT = [0.9216499759579432, -0.38669263218892913, 0.031486742575119436, -0.006222143483426112]


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply [x, y, z, w] quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def grasp_quaternion(banana_heading: float) -> list[float]:
    """Rotate the tool-down pose so the gripper aligns with the banana's long axis."""
    half = banana_heading / 2.0
    yaw_quaternion = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
    quaternion = _quaternion_multiply(yaw_quaternion, DEFAULT_TOOL_QUAT)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


class FrankaPickPlace:

    def __init__(self, executor: MultiThreadedExecutor):
        # The executor allows the robot controller to receive ROS messages while moving
        self.executor = executor
        self.gripper = FrankaGripperController()

        # Create the same Franka velocity controller used by the working reference file
        self.robot = DLSVelocityCommander(
            robot_id='robotB',
            base_link='fr3_link0',
            tip_link='fr3_link8',
            joint_names=[
                'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
                'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
            ],
            target_pos=[0.0, 0.0, 0.0],
            target_quat=[0.0, 0.0, 0.0, 1.0],
            joint_state_topic='/NS_1/franka/joint_states',
            velocity_command_topic='/NS_1/joint_velocity_controller/commands',
            robot_description_topic='/NS_1/robot_description',
            ee_pose_topic=None,
            ee_pose_is_stamped=False,
            max_cartesian_vel=0.17,
            max_angular_vel=0.17,
            dt=0.01,
            damping=0.03,
        )
        self.executor.add_node(self.robot)

    @staticmethod
    def _position(coords: Iterable[float]) -> list[float]:
        position = np.asarray(coords, dtype=np.float64).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError("coords must contain three finite values")

        # Conservative Franka workspace check - rejects bad camera data before any movement
        x, y, z = position
        if not (0.20 <= x <= 0.80 and -0.70 <= y <= 0.45 and 0.05 <= z <= 0.75):
            raise ValueError(f"Target is outside the configured safe workspace: {position.tolist()}")
        return position.tolist()

    def move_and_wait(self, pos: Iterable[float], quat: Iterable[float],
                      name: str, timeout: float = 10.0) -> None:
        """Move to a target and raise an error instead of continuing after a timeout."""
        position = self._position(pos)
        quaternion = np.asarray(quat, dtype=np.float64).reshape(-1)
        if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
            raise ValueError("quat must contain four finite values")
        quaternion /= np.linalg.norm(quaternion)

        self.robot.get_logger().info(f"Moving to {name}")
        self.robot.set_target(position, quaternion.tolist())
        self.robot.reset_goal_reached()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.01)
            if self.robot.goal_reached():
                time.sleep(0.25)
                return

        self.robot.publish_zero_velocity()
        raise TimeoutError(f"Franka timed out while moving to {name}")

    def franka_pick(self, coords: Iterable[float], banana_heading: float,
                    surface_height: float = 0.0) -> None:
        """Approach from above, pick the banana at its midpoint & retreat vertically."""
        pick_pos = np.asarray(self._position(coords))
        quat = grasp_quaternion(banana_heading)

        # Detector sees the banana's upper surface; descend only a limited amount toward its centre
        centre_offset = min(max(surface_height * 0.35, 0.0), 0.025)
        pick_pos[2] = max(0.05, pick_pos[2] - centre_offset)
        # Approach from 12 cm above instead of moving sideways into the banana
        approach_pos = pick_pos.copy()
        approach_pos[2] += 0.12

        # PICK SEQUENCE: open -> approach -> descend -> close -> lift
        self.gripper.open_gripper(width=0.08)
        time.sleep(0.5)
        self.move_and_wait(approach_pos, quat, 'BANANA_APPROACH')
        self.move_and_wait(pick_pos, quat, 'BANANA_PICK')
        self.robot.get_logger().info('Closing gripper')
        self.gripper.close_gripper(width=0.025, force=20.0)
        time.sleep(0.8)
        self.move_and_wait(approach_pos, quat, 'BANANA_RETREAT')

    def franka_place(self, coords: Iterable[float] = DEFAULT_PLACE_POS,
                     quat: Iterable[float] = DEFAULT_PLACE_QUAT) -> None:
        """Move above the cutting board, place the banana & return to rest."""
        place_pos = np.asarray(self._position(coords))
        approach_pos = place_pos.copy()
        approach_pos[2] += 0.12

        # PLACE SEQUENCE: approach -> descend -> open -> lift -> rest
        self.move_and_wait(approach_pos, quat, 'PLACE_APPROACH')
        self.move_and_wait(place_pos, quat, 'PLACE_POSITION')
        self.gripper.open_gripper(width=0.08)
        time.sleep(0.6)
        self.move_and_wait(approach_pos, quat, 'PLACE_RETREAT')
        self.move_and_wait(REST_POS, REST_QUAT, 'REST_POSITION')

    def shutdown(self) -> None:
        """Always leave the velocity controller stopped."""
        self.robot.publish_zero_velocity()
        self.executor.remove_node(self.robot)
        self.robot.destroy_node()


# Module-level wrappers are kept for the interface requested by the original stub
def franka_pick(coords, banana_heading, controller: FrankaPickPlace,
                surface_height: float = 0.0):
    controller.franka_pick(coords, banana_heading, surface_height)


def franka_place(coords, controller: FrankaPickPlace):
    controller.franka_place(coords)
