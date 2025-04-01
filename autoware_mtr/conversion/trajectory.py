from __future__ import annotations
from std_msgs.msg import Header
from rclpy.duration import Duration
from numpy.typing import NDArray
from geometry_msgs.msg import Pose
from autoware_mtr.dataclass.agent import OriginalInfo
# from autoware_perception_msgs.msg import PredictedPath
# from autoware_perception_msgs.msg import PredictedObjects
# from autoware_perception_msgs.msg import PredictedObject
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_new_planning_msgs.msg import Trajectories, TrajectoryGeneratorInfo
from autoware_new_planning_msgs.msg import Trajectory as NewTrajectory
from unique_identifier_msgs.msg import UUID as RosUUID
from std_msgs.msg import String


from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import List, Sequence
from geometry_msgs.msg import Quaternion

import numpy as np

from autoware_mtr.dataclass.agent import AgentState
from awml_pred.ops import rotate_along_z
from tf_transformations import quaternion_from_euler
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def order_from_closest_to_furthest(reference_state: AgentState, histories: List[deque[AgentState]]) -> List[deque[AgentState]]:
    # Sort histories by distance to reference state, making a copy of the list
    return sorted(histories, key=lambda x: np.linalg.norm(x[-1].xyz - reference_state.xyz))


def get_relative_histories(reference_states: List[AgentState], histories: List[deque[AgentState]]) -> List[deque[AgentState]]:
    relative_histories = []

    for n, history in enumerate(histories):
        for b, reference_state in enumerate(reference_states):
            relative_history = get_relative_history(reference_state, history)
            relative_histories.append(relative_history)

    return relative_histories


def get_relative_history(reference_state: AgentState, history: deque[AgentState]) -> deque[AgentState]:
    relative_history = history.copy()
    for i, state in enumerate(history):
        transformed_state_xyz = state.xyz - reference_state.xyz
        transformed_state_xyz[:2] = rotate_along_z(transformed_state_xyz[:2], -reference_state.yaw)
        transformed_state_yaw = state.yaw - reference_state.yaw
        transformed_vxy = rotate_along_z(state.vxy, state.yaw - reference_state.yaw)

        relative_timestamp = state.timestamp - history[0].timestamp
        relative_state = AgentState(uuid=state.uuid, timestamp=relative_timestamp, label_id=state.label_id, xyz=transformed_state_xyz,
                                    size=state.size, yaw=transformed_state_yaw, vxy=transformed_vxy.reshape((2,)), is_valid=state.is_valid)
        relative_history.append(relative_state)
    return relative_history


__all__ = ("to_predicted_objects",)


def to_trajectories(header: Header,
                    infos: Sequence[OriginalInfo],
                    pred_scores: NDArray,
                    pred_trajs: NDArray,
                    score_threshold: float,
                    generator_uuid: RosUUID,
                    ) -> Trajectories:
    """Convert predictions of ego to Trajectory msg.

    Args:
        header (Header): Header of the input message.
        infos (Sequence[OriginalInfo]): List of original message information.
        pred_scores (NDArray): Predicted score tensor in the shape of (N, M).
        pred_trajs (NDArray): Predicted trajectory tensor in the shape of (N, M, T, 4).
        score_threshold (float): Threshold value of score.

    Returns:
        Trajectories: Instanced msg.
    """
    output: Trajectories = Trajectories()
    # convert each object
    info, cur_scores, cur_trajs = infos[0], pred_scores[0], pred_trajs[0]
    output.trajectories = _to_new_trajectories(
        header, info, cur_scores, cur_trajs, score_threshold, generator_uuid)
    generator_name: String = String()
    generator_name.data = "mtr"
    output.generator_info = [TrajectoryGeneratorInfo(
        generator_id=generator_uuid, generator_name=generator_name)]

    return output


def _to_new_trajectories(header: Header,
                         info: OriginalInfo,
                         pred_scores: NDArray,
                         pred_trajs: NDArray,
                         score_threshold: float,
                         generator_uuid: RosUUID,
                         ) -> Trajectories:
    """Convert prediction of a single object to Trajectory msg.

    Args:
        info (ObjectInfo): Object original info.
        pred_scores (NDArray): Predicted score in the shape of (M,).
        pred_trajs (NDArray): Predicted trajectory in the shape of (M, T, 4).
        score_threshold (float): Threshold value of score.

    Returns:
        Trajectory: Instanced msg.
    """

    output: List[Trajectories] = []

    # # convert each mode
    for cur_score, cur_traj in zip(pred_scores, pred_trajs, strict=True):
        if cur_score < score_threshold:
            continue
        cur_mode_traj: NewTrajectory = _to_traj(
            info, cur_traj, cur_score, get_new_trajectory=True)
        cur_mode_traj.header = header
        cur_mode_traj.generator_id = generator_uuid
        output.append(cur_mode_traj)

    return output


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion using ROS2 tf_transformations.

    Args:
        yaw (float): Yaw angle in radians.

    Returns:
        Quaternion: Quaternion representing the yaw angle.
    """
    q = Quaternion()
    q.x, q.y, q.z, q.w = quaternion_from_euler(0, 0, yaw)
    return q


def _to_traj_interp(
    info: OriginalInfo,
    pred_traj: NDArray,
    pred_score: float = 0.0,
    get_new_trajectory: bool = False,
) -> Trajectory:
    """Convert prediction of a single mode to Trajectory msg, with smoothing.

    Args:
        info (OriginalInfo): Object original info.
        pred_traj (NDArray): Predicted waypoints in the shape of (T, 4).
        pred_score (float): Predicted score.
    Returns:
        Trajectory: Smoothed trajectory message.
    """
    output = NewTrajectory() if get_new_trajectory else Trajectory()
    if get_new_trajectory:
        output.score = float(pred_score)

    time_step = 0.1
    T = len(pred_traj)

    # Extract raw x, y, vx, vy
    t = np.linspace(0, (T - 1) * time_step, T)
    x_raw, y_raw, _, _, _, vx_raw, vy_raw = pred_traj.T  # Extract columns

    # Apply cubic spline interpolation
    x_spline = CubicSpline(t, x_raw)
    y_spline = CubicSpline(t, y_raw)

    # Generate smoothed positions
    t_fine = np.linspace(0, (T - 1) * time_step, T)
    x_smooth = x_spline(t_fine)
    y_smooth = y_spline(t_fine)

    # Compute smoothed yaw using dx/dt, dy/dt
    dx = np.gradient(x_smooth, t_fine)
    dy = np.gradient(y_smooth, t_fine)
    yaw_smooth = np.arctan2(dy, dx)

    # Apply Savitzky-Golay filter for extra noise reduction (optional)
    yaw_smooth = savgol_filter(yaw_smooth, window_length=5, polyorder=2)

    for i in range(T):
        pose = Pose()
        pose.position.x = float(x_smooth[i])
        pose.position.y = float(y_smooth[i])
        pose.position.z = info.kinematics.pose_with_covariance.pose.position.z

        pose.orientation = _yaw_to_quaternion(yaw_smooth[i])

        trajectory_point = TrajectoryPoint()
        trajectory_point.pose = pose
        trajectory_point.longitudinal_velocity_mps = float(vx_raw[i])
        trajectory_point.lateral_velocity_mps = float(vy_raw[i])
        trajectory_point.acceleration_mps2 = 0.0
        trajectory_point.time_from_start = Duration(seconds=time_step * i).to_msg()

        output.points.append(trajectory_point)

    return output


def _to_traj(
    info: OriginalInfo,
    pred_traj: NDArray,
    pred_score: float = 0.0,
    get_new_trajectory: bool = False,
) -> Trajectory:
    """Convert prediction of a single mode to Trajectory msg.

    Args:
        info (OriginalInfo): Object original info.
        pred_score (float): Predicted score.
        pred_traj (NDArray): Predicted waypoints in the shape of (T, 4).
    Returns:
        Trajectory: Instanced msg.
    """
    output = NewTrajectory() if get_new_trajectory else Trajectory()
    if get_new_trajectory:
        output.score = float(pred_score)
    time_step = 0.1
    for i, mode_point in enumerate(pred_traj):  # (x, y, vx, vy)
        x, y, _, _, _, vx, vy = mode_point
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = info.kinematics.pose_with_covariance.pose.position.z
        if (i == 0):
            pose.orientation = info.kinematics.pose_with_covariance.pose.orientation
        else:
            prev_x = output.points[i-1].pose.position.x
            prev_y = output.points[i-1].pose.position.y
            yaw = np.arctan2(y-prev_y, x-prev_x)
            pose.orientation = _yaw_to_quaternion(yaw)

        trajectory_point = TrajectoryPoint()
        trajectory_point.pose = pose
        trajectory_point.longitudinal_velocity_mps = float(vx)
        trajectory_point.lateral_velocity_mps = float(vy)
        trajectory_point.acceleration_mps2 = 0.0
        trajectory_point.time_from_start = Duration(seconds=time_step*i).to_msg()
        output.points.append(trajectory_point)

    return output
