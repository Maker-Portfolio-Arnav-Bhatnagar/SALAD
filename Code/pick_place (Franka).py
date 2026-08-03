#Pick Place (Franka):
#Simple code to pick & place a cube using a Franka FR3 robot

#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        max_cartesian_vel=0.17,
        max_angular_vel=0.17,
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

        time.sleep(0.3)

    # ---------------- POSITIONS ----------------

    PICK_POS = [0.6151722180084411, -0.04675817075829586, 0.1507172036929535]
    PICK_QUAT = [0.9208164128617827, -0.38895860442545854, -0.02843004791606157, -0.0002649966960190266]

    POS2 = [0.4287406375189319, -0.30689361698925977, 0.3837058885233575]
    QUAT2 = [0.9184099316444361, -0.39545923684161366, 0.009802773736811583, -0.006252605902784995]

    PLACE_POS = [0.49262495730561356, -0.5186854477752654, 0.16798802875832192]
    PLACE_QUAT = [0.9251516457019573, -0.3795919664139947, 0.0012463626502968118, 0.0016787105351086956]

    REST_POS = [0.3584641619510052, -0.036976077522332576, 0.49373634152165496]
    REST_QUAT = [0.9216499759579432, -0.38669263218892913, 0.031486742575119436, -0.006222143483426112]

    try:

        # ---------- OPEN GRIPPER AT START (JUST TO BE SAFE) ----------
        gripper.open_gripper(width=0.08)

        # ---------- MOVE TO PICK_POS ----------
        move_and_wait(PICK_POS, PICK_QUAT, "PICK_POS")

        # ---------- CLOSE GRIPPER ----------
        robotB.get_logger().info("Closing gripper")
        gripper.close_gripper(width=0.04, force=20.0)
        time.sleep(0.6)

        # ---------- FOLLOW WAYPOINTS ----------
        move_and_wait(POS2, QUAT2, "POS2")
        
        # ---------- MOVE TO PLACE POS ----------
        move_and_wait(PLACE_POS, PLACE_QUAT, "PLACE_POS")

        # ---------- OPEN GRIPPER ----------
        gripper.open_gripper(width=0.08)

        # ---------- MOVE TO REST POS ----------
        move_and_wait(REST_POS, REST_QUAT, "REST_POS")

        robotB.get_logger().info("Sequence complete.")

    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
