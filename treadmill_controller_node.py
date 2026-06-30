import rclpy
from rclpy.node import Node

from treadmill_interfaces.msg import (
    GaitState,
    TreadmillCommand,
)

from treadmill_controller.controller_logic import self_paced


class TreadmillControllerNode(Node):

    def __init__(self):
        super().__init__("treadmill_controller_node")

        self.get_logger().info(
            "Treadmill Controller Node Started"
        )

        # Publish treadmill commands
        self.command_pub = self.create_publisher(
            TreadmillCommand,
            "/treadmill/cmd",
            10
        )

        # Subscribe to processed gait information
        self.gait_sub = self.create_subscription(
            GaitState,
            "/gait/state",
            self.gait_callback,
            10
        )

    def gait_callback(self, msg):

        #
        # TODO:
        #
        # Feed the gait data into self_paced()
        # Once the controller logic returns a desired
        # speed and acceleration, publish a
        # TreadmillCommand.
        #

        pass


def main(args=None):

    rclpy.init(args=args)

    node = TreadmillControllerNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()