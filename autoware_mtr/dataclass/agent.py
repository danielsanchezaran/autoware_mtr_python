from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from typing import Sequence

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import Shape
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjectKinematics
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from unique_identifier_msgs.msg import UUID as RosUUID

__all__ = ("OriginalInfo", "AgentState", "AgentTrajectory")


@dataclass(frozen=True)
class OriginalInfo:
    uuid: RosUUID
    classification: Sequence[ObjectClassification]
    shape: Shape
    existence_probability: float
    kinematics: TrackedObjectKinematics

    @classmethod
    def from_msg(cls, msg: TrackedObject) -> OriginalInfo:
        return cls(
            uuid=msg.object_id,
            classification=msg.classification,
            shape=msg.shape,
            existence_probability=msg.existence_probability,
            kinematics=msg.kinematics,
        )

    @classmethod
    def from_point(cls, point: TrajectoryPoint, uuid: str | RosUUID,
                   dimensions: tuple[float, float, float] | Vector3,
                   ) -> OriginalInfo:
        if not isinstance(uuid, RosUUID):
            uuid = _str_to_uuid_msg(uuid)

        if not isinstance(dimensions, Vector3):
            dimensions = Vector3(x=dimensions[0], y=dimensions[1], z=dimensions[2])
        classification = ObjectClassification()
        classification.label = ObjectClassification.CAR
        classification.probability = 1.0

        kinematics = TrackedObjectKinematics()
        kinematics.pose_with_covariance.pose = point.pose
        kinematics.twist_with_covariance.twist.linear.x = point.longitudinal_velocity_mps
        kinematics.twist_with_covariance.twist.linear.y = point.lateral_velocity_mps

        return cls(
            uuid=uuid,
            classification=[classification],
            shape=Shape(type=0, dimensions=dimensions),
            existence_probability=1.0,
            kinematics=kinematics,
        )

    @classmethod
    def from_odometry(
        cls,
        msg: Odometry,
        uuid: str | RosUUID,
        dimensions: tuple[float, float, float] | Vector3,
    ) -> OriginalInfo:
        if not isinstance(uuid, RosUUID):
            uuid = _str_to_uuid_msg(uuid)

        if not isinstance(dimensions, Vector3):
            dimensions = Vector3(x=dimensions[0], y=dimensions[1], z=dimensions[2])

        classification = ObjectClassification()
        classification.label = ObjectClassification.CAR
        classification.probability = 1.0

        kinematics = TrackedObjectKinematics()
        kinematics.pose_with_covariance = msg.pose
        kinematics.twist_with_covariance = msg.twist
        return cls(
            uuid=uuid,
            classification=[classification],
            shape=Shape(type=0, dimensions=dimensions),
            existence_probability=1.0,
            kinematics=kinematics,
        )


def _str_to_uuid_msg(uuid: str) -> RosUUID:
    """Convert string to ROS uuid msg.

    Args:
        uuid (str): UUID in string it must be 16 length.

    Returns:
        RosUUID: ROS uuid msg.
    """
    uuid_bytes = list(bytes(uuid.encode()))
    return RosUUID(uuid=uuid_bytes)


@dataclass(frozen=True)
class AgentState:
    """A class represents agent state at the specific time."""

    uuid: str
    timestamp: float = -np.inf
    label_id: int = -1
    xyz: NDArray = np.zeros(3)
    size: NDArray = np.zeros(3)
    yaw: float = 0.0
    vxy: NDArray = np.zeros(2)
    is_valid: bool = False

    @property
    def xy(self) -> NDArray:
        return self.xyz[:2]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.xyz[:2] = xy


@dataclass
class AgentTrajectory:
    """
    A class represents agent trajectory.

    Attributes
    ----------
        waypoints (NDArray): Trajectory waypoints in shape (..., D).

    """

    waypoints: NDArray
    label_ids: NDArray

    # NOTE: For the 1DArray indices must be a list.
    XYZ_IDX: ClassVar[list[int]] = [0, 1, 2]
    XY_IDX: ClassVar[list[int]] = [0, 1]
    SIZE_IDX: ClassVar[list[int]] = [3, 4, 5]
    YAW_IDX: ClassVar[int] = 6
    VEL_IDX: ClassVar[list[int]] = [7, 8]
    IS_VALID_IDX: ClassVar[int] = 9

    num_dim: ClassVar[int] = 10

    def __post_init__(self) -> None:
        assert self.waypoints.shape[-1] == self.num_dim
        assert len(self.waypoints) == len(self.label_ids)

    @property
    def xyz(self) -> NDArray:
        return self.waypoints[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: ArrayLike) -> None:
        self.waypoints[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        return self.waypoints[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.waypoints[..., self.XY_IDX] = xy

    @property
    def size(self) -> NDArray:
        return self.waypoints[..., self.SIZE_IDX]

    @size.setter
    def size(self, size: ArrayLike) -> None:
        self.waypoints[..., self.SIZE_IDX] = size

    @property
    def yaw(self) -> NDArray:
        return self.waypoints[..., self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw: ArrayLike) -> None:
        self.waypoints[..., self.YAW_IDX] = yaw

    @property
    def vxy(self) -> NDArray:
        return self.waypoints[..., self.VEL_IDX]

    @vxy.setter
    def vxy(self, velocity: ArrayLike) -> None:
        self.waypoints[..., self.VEL_IDX] = velocity

    @property
    def is_valid(self) -> NDArray:
        return self.waypoints[..., self.IS_VALID_IDX] == 1

    @property
    def shape(self) -> ArrayLike:
        return self.waypoints.shape

    def as_array(self) -> NDArray:
        return self.waypoints.copy()
