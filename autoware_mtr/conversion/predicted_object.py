from typing import Sequence

from autoware_perception_msgs.msg import PredictedObject
from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import PredictedPath
from autoware_mtr.dataclass.agent import OriginalInfo
from geometry_msgs.msg import Pose
from numpy.typing import NDArray
from rclpy.duration import Duration
from std_msgs.msg import Header

__all__ = ("to_predicted_objects",)


def to_predicted_objects(
    header: Header,
    infos: Sequence[OriginalInfo],
    pred_scores: NDArray,
    pred_trajs: NDArray,
    score_threshold: float,
) -> PredictedObjects:
    """Convert predictions to PredictedObjects msg.

    Args:
        header (Header): Header of the input message.
        infos (Sequence[OriginalInfo]): List of original message information.
        pred_scores (NDArray): Predicted score tensor in the shape of (N, M).
        pred_trajs (NDArray): Predicted trajectory tensor in the shape of (N, M, T, 7).
        score_threshold (float): Threshold value of score.

    Returns:
        PredictedObjects: Instanced msg.
    """
    output = PredictedObjects()
    output.header = header
    # convert each object
    for info, cur_scores, cur_trajs in zip(infos, pred_scores, pred_trajs, strict=True):
        pred_obj = _to_predicted_object(info, cur_scores, cur_trajs, score_threshold)
        output.objects.append(pred_obj)

    return output


def _to_predicted_object(
    info: OriginalInfo,
    pred_scores: NDArray,
    pred_trajs: NDArray,
    score_threshold: float,
) -> PredictedObject:
    """Convert prediction of a single object to PredictedObject msg.

    Args:
        info (ObjectInfo): Object original info.
        pred_scores (NDArray): Predicted score in the shape of (M,).
        pred_trajs (NDArray): Predicted trajectory in the shape of (M, T, 4).
        score_threshold (float): Threshold value of score.

    Returns:
        PredictedObject: Instanced msg.
    """
    output = PredictedObject()

    output.object_id = info.uuid
    output.classification = info.classification
    output.shape = info.shape
    output.existence_probability = info.existence_probability

    output.kinematics.initial_pose_with_covariance = info.kinematics.pose_with_covariance
    output.kinematics.initial_twist_with_covariance = info.kinematics.twist_with_covariance
    output.kinematics.initial_acceleration_with_covariance = (
        info.kinematics.acceleration_with_covariance
    )

    # convert each mode
    for cur_score, cur_traj in zip(pred_scores, pred_trajs, strict=True):
        if cur_score < score_threshold:
            continue
        cur_mode_path = _to_predicted_path(info, cur_score, cur_traj)
        output.kinematics.predicted_paths.append(cur_mode_path)

    return output


def _to_predicted_path(
    info: OriginalInfo,
    pred_score: float,
    pred_traj: NDArray,
) -> PredictedPath:
    """Convert prediction of a single mode to PredictedPath msg.

    Args:
        info (OriginalInfo): Object original info.
        pred_score (float): Predicted score.
        pred_traj (NDArray): Predicted waypoints in the shape of (T, 4).

    Returns:
        PredictedPath: Instanced msg.
    """
    output = PredictedPath()
    output.time_step = Duration(seconds=0.1).to_msg()  # TODO(ktro2828): use specific value?
    output.confidence = float(pred_score)
    for x, y, _, _, _, _, _ in pred_traj:  # (x, y, vx, vy)
        pose = Pose()

        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = info.kinematics.pose_with_covariance.pose.position.z
        pose.orientation = info.kinematics.pose_with_covariance.pose.orientation

        output.path.append(pose)

    return output
