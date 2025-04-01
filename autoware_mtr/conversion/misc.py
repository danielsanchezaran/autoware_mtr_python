from geometry_msgs.msg import Quaternion as RosQuaternion
from pyquaternion import Quaternion as PyQuaternion
from std_msgs.msg import Header

__all__ = ("timestamp2ms", "yaw_from_quaternion")


def timestamp2ms(header: Header) -> float:
    """Convert ROS timestamp to milliseconds.

    Args:
        header (Header): ROS msg header.

    Returns:
        float: Timestamp in [ms].
    """
    return header.stamp.sec * 1e3 + header.stamp.nanosec * 1e-6


def timestamp2us(header: Header) -> float:
    """Convert ROS timestamp to microseconds.

    Args:
        header (Header): ROS msg header.

    Returns:
        float: Timestamp in [us].
    """
    return header.stamp.sec * 1e6 + header.stamp.nanosec * 1e-3


def yaw_from_quaternion(orientation: RosQuaternion) -> float:
    """Convert ROS msg quaternion to yaw angle.

    Args:
        orientation (RosQuaternion): ROS msg quaternion.

    Returns:
        float: Yaw angle in [rad].
    """
    yaw, _, _ = PyQuaternion(
        orientation.w, orientation.x, orientation.y, orientation.z
    ).yaw_pitch_roll
    return yaw
