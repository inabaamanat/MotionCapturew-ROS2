import rclpy
from rclpy.node import Node

# Custom messages we created in treadmill_interfaces
from treadmill_interfaces.msg import TreadmillCommand
from treadmill_interfaces.msg import TreadmillState

# Hardware API (contains all TCP communication)
from treadmill_driver.treadmill_api import (
    read_treadmill,
    set_treadmill
)


class TreadmillDriverNode(Node):

    def __init__(self):
        super().__init__("treadmill_driver_node")

        self.get_logger().info(
            "Treadmill Driver Node Started"
        )

        # ----------------------------------------------------------
        # Publisher
        #
        # Publishes the CURRENT treadmill state.
        #
        # Other nodes (gait, controller, GUI, etc.) can subscribe
        # to this topic instead of directly talking to the treadmill.
        # ----------------------------------------------------------
        self.state_pub = self.create_publisher(
            TreadmillState,
            "/treadmill/state",
            10
        )

        # ----------------------------------------------------------
        # Subscriber
        #
        # Listens for treadmill speed commands.
        #
        # Eventually the controller node will publish these.
        # ----------------------------------------------------------
        self.command_sub = self.create_subscription(
            TreadmillCommand,
            "/treadmill/cmd",
            self.command_callback,
            10
        )

        # ----------------------------------------------------------
        # Timer
        #
        # Every 0.1 seconds (10 Hz), read the treadmill state and
        # publish it to the rest of the ROS system.
        # ----------------------------------------------------------
        self.timer = self.create_timer(
            0.1,
            self.publish_state
        )

    # --------------------------------------------------------------
    # Called automatically whenever another ROS node publishes a
    # TreadmillCommand message.
    # --------------------------------------------------------------
    # --------------------------------------------------------------
# Called automatically whenever another ROS node publishes a
# TreadmillCommand message.
# --------------------------------------------------------------
def command_callback(self, msg):

    try:

        # ----------------------------------------------------------
        # Send the requested command to the treadmill.
        #
        # We pass every field from the ROS message directly to the
        # hardware API so the driver stays as a thin translation
        # layer between ROS and the Bertec treadmill.
        # ----------------------------------------------------------
        set_treadmill(
            msg.right_speed,
            msg.left_speed,
            msg.right_acceleration,
            msg.left_acceleration,
            msg.incline
        )

    except Exception as e:

        self.get_logger().error(
            f"Failed to send treadmill command: {e}"
        )

    # --------------------------------------------------------------
    # Called every 0.1 seconds by the timer.
    #
    # Reads the treadmill's current state and publishes it so every
    # other ROS node knows the current speed and incline.
    # --------------------------------------------------------------
    def publish_state(self):

        try:

            speed_r, speed_l, incline = read_treadmill()

            state_msg = TreadmillState()

            state_msg.right_speed = float(speed_r)
            state_msg.left_speed = float(speed_l)
            state_msg.incline = float(incline)

            self.state_pub.publish(state_msg)

        except Exception as e:

            self.get_logger().error(
                f"Treadmill connection failed: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = TreadmillDriverNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()