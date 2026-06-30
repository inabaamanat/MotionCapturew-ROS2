import rclpy
from rclpy.node import Node

from treadmill_interfaces.msg import TreadmillCommand


class TestCommandNode(Node):

    def __init__(self):
        super().__init__("test_command_node")

        # Publish treadmill commands
        self.publisher = self.create_publisher(
            TreadmillCommand,
            "/treadmill/cmd",
            10
        )

        # Send a command every second
        self.timer = self.create_timer(
            1.0,
            self.publish_command
        )

        self.get_logger().info("Test Command Node Started")

    def publish_command(self):

        msg = TreadmillCommand()

        # ---------------------------------------------------
        # Test values
        #
        # These are intentionally small and safe.
        # Once we're in the lab we can adjust them.
        # ---------------------------------------------------

        msg.right_speed = 500.0
        msg.left_speed = 500.0

        msg.right_acceleration = 500.0
        msg.left_acceleration = 500.0

        msg.incline = 0.0

        self.publisher.publish(msg)

        self.get_logger().info(
            "Published treadmill command."
        )


def main(args=None):

    rclpy.init(args=args)

    node = TestCommandNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()