from autoware_mtr.dataclass.agent import AgentState
from autoware_mtr.dataclass.agent import OriginalInfo
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from std_msgs.msg import Header
import numpy as np
from numpy.typing import NDArray

from .misc import timestamp2ms
from .misc import yaw_from_quaternion

__all__ = ("from_odometry", "convert_transform_stamped")


def from_trajectory_point(point: TrajectoryPoint, uuid: str, header: Header, label_id: int, size: NDArray) -> tuple[AgentState, OriginalInfo]:
    timestamp = timestamp2ms(header=header) * 1e-3  # Requires microseconds
    pose = point.pose
    xyz = np.array((pose.position.x, pose.position.y, pose.position.z))

    yaw = yaw_from_quaternion(pose.orientation)

    twist_x = point.longitudinal_velocity_mps
    twist_y = point.lateral_velocity_mps

    vxy = np.array((twist_x, twist_y))

    state = AgentState(
        uuid=uuid,
        timestamp=timestamp,
        label_id=label_id,
        xyz=xyz,
        size=size,
        yaw=yaw,
        vxy=vxy,
        is_valid=True,
    )

    info = OriginalInfo.from_point(point, uuid, size)

    return state, info


def from_odometry(
    msg: Odometry,
    uuid: str,
    label_id: int,
    size: NDArray,
) -> tuple[AgentState, OriginalInfo]:
    """Convert odometry msg to AgentState.

    Args:
        msg (Odometry): Odometry msg.
        uuid (str): Object uuid.
        label_id (int): Label id.
        size (NDArray): Object size in the order of (length, width, height).

    Returns:
        tuple[AgentState, OriginalInfo]: Instanced AgentState.
    """
    timestamp = timestamp2ms(msg.header)
    pose = msg.pose.pose
    xyz = np.array((pose.position.x, pose.position.y, pose.position.z))

    yaw = yaw_from_quaternion(pose.orientation)

    twist = msg.twist.twist
    vxy = np.array((twist.linear.x, twist.linear.y))

    state = AgentState(
        uuid=uuid,
        timestamp=timestamp,
        label_id=label_id,
        xyz=xyz,
        size=size,
        yaw=yaw,
        vxy=vxy,
        is_valid=True,
    )

    info = OriginalInfo.from_odometry(msg, uuid=uuid, dimensions=size)

    return state, info


def convert_transform_stamped(
    tf_stamped: TransformStamped,
    uuid: str,
    label_id: int,
    size: NDArray,
    vxy: NDArray,
) -> AgentState:
    timestamp = timestamp2ms(tf_stamped.header)

    translation = tf_stamped.transform.translation
    xyz = np.array((translation.x, translation.y, translation.z))
    yaw = yaw_from_quaternion(tf_stamped.transform.rotation)

    return AgentState(
        uuid=uuid,
        timestamp=timestamp,
        label_id=label_id,
        xyz=xyz,
        size=size,
        yaw=yaw,
        vxy=vxy,
        is_valid=True,
    )
