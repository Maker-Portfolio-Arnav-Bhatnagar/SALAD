# SALAD_V1.py:
# Detects the banana midpoint & orientation, converts them to Franka coordinates, then performs pick/place
# Stops after placing the banana because vegetable_cutter.py is not implemented yet

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from banana_detector import BananaDetector
from coordinate_transformer import transform_object_angle, transform_point
from pick_place import DEFAULT_PLACE_POS, FrankaPickPlace

DETECTION_TIMEOUT = 20.0

def main(args=None):
    # Start ROS and create one executor to process camera and robot messages
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=3)
    detector = BananaDetector(show_debug=True)
    executor.add_node(detector)
    motion = None

    try:
        # STEP 1: Wait until the detector sees the same banana for several frames
        detector.get_logger().info('Waiting for a stable banana detection')
        detection = detector.wait_for_detection(executor, timeout=DETECTION_TIMEOUT, stable_frames=3)

        # STEP 2: Convert the camera result into the Franka robot's coordinate frame
        pick_pos = transform_point(detection.midpoint_camera).tolist()
        banana_heading = transform_object_angle(detection.angle_camera)
        detector.get_logger().info(
            f"Stable banana found | Franka midpoint: {pick_pos} | "
            f"heading: {banana_heading:.3f} rad | height: {detection.surface_height:.3f} m"
        )

        # STEP 3: Stop camera processing, then pick up and place the banana
        executor.remove_node(detector)
        motion = FrankaPickPlace(executor)
        motion.franka_pick(pick_pos, banana_heading, detection.surface_height)
        motion.franka_place(DEFAULT_PLACE_POS)

        # STEP 4: The vegetable cutter will be called here once that file is ready
        motion.robot.get_logger().info(
            'Pick/place complete. Stopping before the vegetable_cutter.py stage.'
        )

    except (KeyboardInterrupt, TimeoutError):
        detector.get_logger().warn('SALAD sequence stopped before completion')
    except Exception as error:
        detector.get_logger().error(f'SALAD sequence failed safely: {error}')
        raise
    finally:
        # This section runs after success, an error or Ctrl+C
        # It stops robot velocity commands and shuts every ROS node down cleanly
        if motion is not None:
            motion.shutdown()
        try:
            executor.remove_node(detector)
        except Exception:
            pass
        detector.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
