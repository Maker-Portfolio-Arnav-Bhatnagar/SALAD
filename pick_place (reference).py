#!/usr/bin/env python3
from _future_ import annotations
import os, sys, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# -------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    gripper = FrankaGripperController()
    gripper.open_gripper(width=0.08)
    time.sleep(1.0)

    robotB = DLSVelocityCommander(
        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=[
            "fr3_joint1","fr3_joint2","fr3_joint3",
            "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7",
        ],
        target_pos=[0.0,0.0,0.0],
        target_quat=[0.0,0.0,0.0,1.0],
        joint_state_topic="/NS_1/franka/joint_states",
        velocity_command_topic="/NS_1/joint_velocity_controller/commands",
        robot_description_topic="/NS_1/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.25,
        max_angular_vel=0.25,
        dt=0.01,
        damping=0.03,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(robotB)

    # -------------------------------------------------------
    def move_and_wait(pos, quat, name, timeout=8.0):
        robotB.get_logger().info(f"Moving to {name}")
        robotB.set_target(pos, quat)
        robotB.reset_goal_reached()

        start = time.time()
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)

            if robotB.goal_reached():
                break

            if time.time() - start > timeout:
                robotB.get_logger().warn(f"Timeout at {name}")
                break

        robotB.publish_zero_velocity()
        time.sleep(0.3)

    # ---------------- POSITIONS ----------------

    POS1 = [0.45251871374475783, -0.02033126129394388, 0.17926676018161913]
    QUAT1 = [0.9103652577315129, -0.4128565467403135,
             -0.02628048398543751, 0.009690484538489988]

    POS2 = [0.4530152262884078, -0.011781334955463256, 0.18222219588404157]
    QUAT2 = [0.9130839393211533, -0.4058866408766574,
             -0.011638397714697735, 0.037393879315502476]

    POS3 = [0.4519261174767757, -0.033347772163313784, 0.1847477097679242]
    QUAT3 = [0.9059893381151105, -0.4195046612530423,
             -0.04685933767824142, -0.03167587222272914]

    POS4 = [0.4492033867409548, -0.01029258178553643, 0.18622842929577677]
    QUAT4 = [0.9128259365568111, -0.40632218554719585,
             -0.008437769006119795, 0.03974789473079201]

    POS5 = [0.4452253351459746, -0.016027448402670317, 0.20099818262036315]
    QUAT5 = [0.9097573840666969, -0.4143783570781459,
             -0.0022063186953863393, 0.025044190526979176]

    try:

        # ---------- MOVE TO POS1 ----------
        move_and_wait(POS1, QUAT1, "POS1")

        # ---------- CLOSE GRIPPER ----------
        robotB.get_logger().info("Closing gripper")
        gripper.close_gripper(width=0.04, force=20.0)
        time.sleep(0.6)

        # ---------- FOLLOW WAYPOINTS ----------
        move_and_wait(POS2, QUAT2, "POS2")
        move_and_wait(POS3, QUAT3, "POS3")
        move_and_wait(POS4, QUAT4, "POS4")
        move_and_wait(POS5, QUAT5, "POS5")

        robotB.get_logger().info("Sequence complete.")

    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        rclpy.shutdown()


if _name_ == "_main_":
    main()