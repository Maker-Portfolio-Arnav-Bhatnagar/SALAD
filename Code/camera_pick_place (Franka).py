from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

class object_detect(Node):

    def __init__(self):

        # Initialize CV Bridge
        self._bridge = CvBridge()

        # Create subscriptions
        image_sub = self.create_subscription(
                    ROSImage,
                    '/camera/camera/color/image_raw',
                    self._image_callback,
                    10
                )
        depth_sub = self.create_subscription(
                    ROSImage,
                    '/camera/camera/aligned_depth_to_color/image_raw',
                    self._depth_callback,
                    10
                )
        def _image_callback(self, msg)-> None:
            """Store latest color image."""
            self._color_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")

        def _depth_callback(self, msg)-> None:
            """Store latest depth image."""
            self._depth_image = self._bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3
