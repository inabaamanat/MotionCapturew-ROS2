import rclpy
from rclpy.node import Node

from treadmill_interfaces.msg import ForcePlateData

from treadmill_daq.daq_logic import collect_data


class TreadmillDAQNode(Node):

    def __init__(self):
        super().__init__("treadmill_daq_node")

        self.get_logger().info(
            "Treadmill DAQ Node Started"
        )

        # Publish raw force plate data
        self.publisher = self.create_publisher(
            GaitState,
            "/daq/raw",
            10
        )

        # todo:
        # Start collect_data()
        # Publish each frame to /daq/raw


def main(args=None):

    rclpy.init(args=args)

    node = TreadmillDAQNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()