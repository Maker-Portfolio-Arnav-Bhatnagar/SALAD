# banana_cutter.py:
# Calculates evenly spaced cut points from the banana's 4 bounding-box corners
# Moves the Franka knife back & forth while slowly descending at each cut point
# Stops descending when the Franka's built-in force estimate detects the cutting board

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from pick_place import move_and_wait


# -----------------------------------------------------------------------------
# VALUES TO CHECK ON THE REAL ROBOT
# -----------------------------------------------------------------------------

# This is the built-in Franka external-wrench topic, not an external sensor
# Confirm it using: ros2 topic list | grep wrench
WRENCH_TOPIC = '/NS_1/franka_robot_state_broadcaster/stiffness_frame_wrench'

# The knife is assumed to point downward along its local Z axis
FORCE_AXIS = 'z'

# Contact must exceed this change from the stationary baseline
# Begin testing with no knife/banana and tune this value carefully
CONTACT_FORCE_THRESHOLD = 6.0  # Newtons
CONTACT_SAMPLES_REQUIRED = 5

# Cutting geometry
SLICE_SPACING = 0.020       # 20 mm between cuts
END_MARGIN = 0.015          # Do not cut the first/last 15 mm of the banana
APPROACH_HEIGHT = 0.050     # Begin each cut 50 mm above the detected banana
RETREAT_HEIGHT = 0.080      # Lift 80 mm after touching the board
SAW_STROKE = 0.025          # Total side-to-side knife travel
DESCENT_PER_STROKE = 0.002  # Descend 2 mm after each half-stroke
MAXIMUM_DESCENT = 0.120     # Never descend more than 120 mm during one cut
MOTION_TIMEOUT = 8.0


@dataclass
class CutPlan:
    # Each cut point is [x, y, z] in the Franka base frame
    cut_points: list[list[float]]

    # Unit vector describing the direction of the back-and-forth sawing movement
    saw_direction: list[float]


def calculate_cut_plan(corners, slice_spacing=SLICE_SPACING,
                       end_margin=END_MARGIN):
    """Calculate cut points and sawing direction from 4 robot-frame corners."""
    corners = np.asarray(corners, dtype=np.float64)

    if corners.shape != (4, 3):
        raise ValueError('corners must contain exactly four [x, y, z] points')
    if not np.all(np.isfinite(corners)):
        raise ValueError('corners contains an invalid number')
    if slice_spacing <= 0.0:
        raise ValueError('slice_spacing must be greater than zero')
    if end_margin < 0.0:
        raise ValueError('end_margin cannot be negative')

    # The detector returns rectangle corners in order around the box
    # Measure all four sides to find which pair represents the banana's length
    side_vectors = np.roll(corners, -1, axis=0) - corners
    side_lengths = np.linalg.norm(side_vectors[:, :2], axis=1)
    longest_side_index = int(np.argmax(side_lengths))

    # The long side joins corner i to corner i+1
    # The opposite long side joins the other two corners
    i = longest_side_index
    start_a = corners[i]
    end_a = corners[(i + 1) % 4]
    start_b = corners[(i - 1) % 4]
    end_b = corners[(i + 2) % 4]

    # Midpoints of the two short ends give the banana's centre line
    line_start = (start_a + start_b) / 2.0
    line_end = (end_a + end_b) / 2.0
    long_vector = line_end - line_start
    banana_length = np.linalg.norm(long_vector[:2])

    if banana_length < 0.001:
        raise ValueError('banana bounding box is too short to calculate cuts')
    if 2.0 * end_margin >= banana_length:
        raise ValueError('end_margin leaves no room for any cuts')

    long_direction = long_vector / np.linalg.norm(long_vector)

    # Saw across the banana, perpendicular to its long direction
    saw_direction = np.array([
        -long_direction[1],
        long_direction[0],
        0.0,
    ])
    saw_direction /= np.linalg.norm(saw_direction)

    # Start after the first margin and stop before the final margin
    usable_length = banana_length - 2.0 * end_margin
    number_of_cuts = int(np.floor(usable_length / slice_spacing))

    cut_points = []
    for cut_number in range(number_of_cuts):
        distance = end_margin + (cut_number + 1) * slice_spacing

        # Do not allow the final cut to enter the far end margin
        if distance >= banana_length - end_margin:
            break

        cut_point = line_start + distance * long_direction
        cut_points.append(cut_point.tolist())

    if len(cut_points) == 0:
        raise ValueError('banana is too short for the requested slice spacing')

    return CutPlan(
        cut_points=cut_points,
        saw_direction=saw_direction.tolist(),
    )


class FrankaForceMonitor(Node):
    """Reads the Franka's built-in estimate of force at the end effector."""

    def __init__(self, wrench_topic=WRENCH_TOPIC):
        super().__init__('franka_knife_force_monitor')

        self.latest_force = None
        self.baseline_force = 0.0
        self.contact_count = 0
        self.sample_number = 0
        self.last_checked_sample = 0
        self.baseline_readings = []

        # Franka convenience topics use best-effort QoS in current franka_ros2
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            wrench_topic,
            self.wrench_callback,
            qos,
        )

    def wrench_callback(self, msg):
        """Store the newest force measured along the configured knife axis."""
        force = msg.wrench.force
        self.latest_force = float(getattr(force, FORCE_AXIS))
        self.sample_number += 1
        self.baseline_readings.append(self.latest_force)
        self.baseline_readings = self.baseline_readings[-100:]

    def reset_contact(self):
        """Forget contact samples from the previous cut."""
        self.contact_count = 0
        self.last_checked_sample = self.sample_number

    def contact_detected(self):
        """Return True only after several consecutive high-force readings."""
        if self.latest_force is None:
            self.contact_count = 0
            return False

        # Do not count the same wrench message more than once
        if self.sample_number == self.last_checked_sample:
            return False
        self.last_checked_sample = self.sample_number

        force_change = abs(self.latest_force - self.baseline_force)

        if force_change >= CONTACT_FORCE_THRESHOLD:
            self.contact_count += 1
        else:
            self.contact_count = 0

        return self.contact_count >= CONTACT_SAMPLES_REQUIRED

    def calibrate_baseline(self, executor, samples=50, timeout=3.0):
        """Average stationary force readings before the knife starts descending."""
        self.baseline_readings = []
        deadline = time.time() + timeout

        while rclpy.ok() and time.time() < deadline:
            executor.spin_once(timeout_sec=0.02)
            if len(self.baseline_readings) >= samples:
                break

        if len(self.baseline_readings) < samples:
            raise TimeoutError(
                f'Only received {len(self.baseline_readings)} of '
                f'{samples} required wrench samples'
            )

        self.baseline_force = float(np.mean(self.baseline_readings[:samples]))
        self.reset_contact()
        self.get_logger().info(
            f'Knife force baseline: {self.baseline_force:.2f} N'
        )


class BananaCutter:
    """Uses an existing Franka controller to carry out all planned cuts."""

    def __init__(self, robot, executor, knife_quat,
                 wrench_topic=WRENCH_TOPIC):
        self.robot = robot
        self.executor = executor
        self.knife_quat = list(knife_quat)

        # Add the force-monitor node to the same executor as the robot
        self.force_monitor = FrankaForceMonitor(wrench_topic)
        self.executor.add_node(self.force_monitor)

    def move_until_contact(self, position, name, timeout=MOTION_TIMEOUT):
        """Move toward one saw endpoint, stopping immediately after contact."""
        self.robot.get_logger().info(f'Moving to {name}')
        self.robot.set_target(position, self.knife_quat)
        self.robot.reset_goal_reached()
        start = time.time()

        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0.01)

            # Stop sending motion as soon as board contact is confirmed
            if self.force_monitor.contact_detected():
                self.robot.publish_zero_velocity()
                self.robot.get_logger().info('Cutting-board contact detected')
                return True

            if self.robot.goal_reached():
                return False

            if time.time() - start > timeout:
                self.robot.publish_zero_velocity()
                raise TimeoutError(f'Timeout while moving to {name}')

        return False

    def cut_at_point(self, cut_point, saw_direction, cut_number):
        """Perform one descending back-and-forth cut at a supplied point."""
        cut_point = np.asarray(cut_point, dtype=np.float64)
        saw_direction = np.asarray(saw_direction, dtype=np.float64)

        # First move above the banana without checking for board contact
        approach = cut_point.copy()
        approach[2] += APPROACH_HEIGHT
        move_and_wait(
            self.robot,
            self.executor,
            approach.tolist(),
            self.knife_quat,
            f'CUT_{cut_number}_APPROACH',
        )

        # Measure the force created by the held knife before descending
        self.force_monitor.calibrate_baseline(self.executor)

        current_height = float(approach[2])
        lowest_allowed_height = current_height - MAXIMUM_DESCENT
        move_to_positive_side = True

        while current_height > lowest_allowed_height:
            # Alternate between equal distances on either side of the cut point
            direction_sign = 1.0 if move_to_positive_side else -1.0
            saw_offset = direction_sign * (SAW_STROKE / 2.0) * saw_direction

            # Every half-stroke moves the knife slightly lower
            current_height -= DESCENT_PER_STROKE
            target = cut_point + saw_offset
            target[2] = current_height

            touched_board = self.move_until_contact(
                target.tolist(),
                f'CUT_{cut_number}_SAW',
            )
            if touched_board:
                break

            move_to_positive_side = not move_to_positive_side
        else:
            self.robot.publish_zero_velocity()
            raise RuntimeError(
                f'Cut {cut_number} reached maximum descent without detecting the board'
            )

        # Lift vertically before travelling to the next cut point
        retreat = cut_point.copy()
        retreat[2] = current_height + RETREAT_HEIGHT
        move_and_wait(
            self.robot,
            self.executor,
            retreat.tolist(),
            self.knife_quat,
            f'CUT_{cut_number}_RETREAT',
        )

    def cut_banana(self, corners, slice_spacing=SLICE_SPACING,
                   end_margin=END_MARGIN):
        """Calculate and execute all cuts across the banana."""
        plan = calculate_cut_plan(corners, slice_spacing, end_margin)
        self.robot.get_logger().info(
            f'Calculated {len(plan.cut_points)} banana cut points'
        )

        for index, cut_point in enumerate(plan.cut_points, start=1):
            self.robot.get_logger().info(
                f'Starting cut {index} of {len(plan.cut_points)}'
            )
            self.force_monitor.reset_contact()
            self.cut_at_point(cut_point, plan.saw_direction, index)

        self.robot.publish_zero_velocity()
        self.robot.get_logger().info('All banana cuts complete')
        return plan.cut_points

    def shutdown(self):
        """Remove the force-monitor node after cutting is finished."""
        self.robot.publish_zero_velocity()
        self.executor.remove_node(self.force_monitor)
        self.force_monitor.destroy_node()


# FLOWCHART:
#
# Receive 4 banana bounding-box corners in Franka coordinates
#                         |
#                         v
# Find banana centre line & calculate evenly spaced cut points
#                         |
#                         v
# Move knife above the first cut point
#                         |
#                         v
# Measure stationary force baseline
#                         |
#                         v
# Saw left/right while descending 2 mm per half-stroke
#                         |
#                  Board contact detected?
#                    /                 \
#                  No                   Yes
#                  |                     |
#          Continue sawing down       Stop velocity
#                                        |
#                                        v
#                              Lift above the banana
#                                        |
#                                        v
#                              Move to the next cut point
