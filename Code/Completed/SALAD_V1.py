# SALAD_V1.py:
# Detects the carrot midpoint & orientation, converts them to Franka coordinates, then performs pick/place
# Stops after placing the carrot because vegetable_cutter.py is not implemented yet

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from carrot_detector import CarrotDetector
from coordinate_transformer import transform_carrot_angle, transform_point
from pick_place import DEFAULT_PLACE_POS, FrankaPickPlace


DETECTION_TIMEOUT = 20.0


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=3)
    detector = CarrotDetector(show_debug=True)
    executor.add_node(detector)
    motion = None

    try:
        detector.get_logger().info('Waiting for a stable carrot detection')
        detection = detector.wait_for_detection(executor, timeout=DETECTION_TIMEOUT)

        # Transform midpoint & long-axis orientation into the Franka base frame
        pick_pos = transform_point(detection.midpoint_camera).tolist()
        carrot_heading = transform_carrot_angle(detection.angle_camera)
        detector.get_logger().info(
            f"Stable carrot found | Franka midpoint: {pick_pos} | "
            f"heading: {carrot_heading:.3f} rad | height: {detection.surface_height:.3f} m"
        )

        # Stop processing camera frames before robot motion begins
        executor.remove_node(detector)
        motion = FrankaPickPlace(executor)
        motion.franka_pick(pick_pos, carrot_heading, detection.surface_height)
        motion.franka_place(DEFAULT_PLACE_POS)

        motion.robot.get_logger().info(
            'Pick/place complete. Stopping before the vegetable_cutter.py stage.'
        )

    except (KeyboardInterrupt, TimeoutError):
        detector.get_logger().warn('SALAD sequence stopped before completion')
    except Exception as error:
        detector.get_logger().error(f'SALAD sequence failed safely: {error}')
        raise
    finally:
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
