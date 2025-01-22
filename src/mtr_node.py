import hashlib
import torch
import numpy as np
from collections import deque

import rclpy
import rclpy.duration
import rclpy.parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from nav_msgs.msg import Odometry

from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor
from utils.polyline import TargetCentricPolyline

from autoware_perception_msgs.msg import PredictedObjects
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_new_planning_msgs.msg import Trajectories
from unique_identifier_msgs.msg import UUID as RosUUID
from autoware_mtr.dataclass.agent import _str_to_uuid_msg

from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjects

from awml_pred.dataclass import AWMLStaticMap
from awml_pred.common import Config, load_checkpoint
from awml_pred.models import build_model
from awml_pred.deploy.apis.torch2onnx import _load_inputs
from utils.lanelet_converter import convert_lanelet
from utils.load import LoadIntentionPoint
from autoware_mtr.conversion.ego import from_odometry
from autoware_mtr.conversion.tracked_object import from_tracked_objects
from autoware_mtr.conversion.misc import timestamp2ms
from autoware_mtr.conversion.trajectory import get_relative_histories, order_from_closest_to_furthest, to_trajectory, to_trajectories
from autoware_mtr.datatype import AgentLabel
from autoware_mtr.geometry import rotate_along_z
from autoware_mtr.dataclass.history import AgentHistory
from autoware_mtr.dataclass.agent import AgentState
from autoware_mtr.conversion.predicted_object import to_predicted_objects
from typing import List


def softmax(x: NDArray, axis: int) -> NDArray:
    """Apply softmax.

    Args:
        x (NDArray): Input array.
        axis (int): Axis to apply softmax.

    Returns:
        NDArray: Softmax result.
    """
    x -= x.max(axis=axis, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=axis, keepdims=True)


class MTRNode(Node):
    def __init__(self) -> None:
        super().__init__("mtr_python_node")

        # subscribers

        qos_profile_2 = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._tracked_objects_sub = self.create_subscription(
            TrackedObjects, "~/input/tracked_objects", self._tracked_objects_callback, qos_profile_2)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._subscription = self.create_subscription(
            Odometry,
            "~/input/ego",
            self._callback,
            qos_profile,
        )

        # ROS parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)

        build_only = (
            self.declare_parameter("build_only", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        model_config_path = (
            self.declare_parameter("model_config", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        deploy_config_path = (
            self.declare_parameter("deploy_config", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        checkpoint_path = (
            self.declare_parameter("checkpoint_path", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        num_timestamp = (
            self.declare_parameter("num_timestamp", descriptor=descriptor)
            .get_parameter_value()
            .integer_value
        )

        self._timestamp_threshold = (
            self.declare_parameter("timestamp_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )

        self._score_threshold = (
            self.declare_parameter("score_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )

        lanelet_file = (
            self.declare_parameter("lanelet_file", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        labels = (
            self.declare_parameter("labels", descriptor=descriptor)
            .get_parameter_value()
            .string_array_value
        )

        intention_point_file = (
            self.declare_parameter("intention_point_file", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        self.ego_dimensions = (self.declare_parameter(
            "ego_dimensions", descriptor=descriptor).get_parameter_value().double_array_value)

        self._num_timestamps = num_timestamp
        self._history = AgentHistory(max_length=num_timestamp)
        self._awml_static_map: AWMLStaticMap = convert_lanelet(lanelet_file)

        intention_point_loader: LoadIntentionPoint = LoadIntentionPoint(
            intention_point_file, labels)
        self._intention_points = intention_point_loader()

        num_polylines: int = 768
        num_points: int = 20
        break_distance: float = 1.0
        center_offset: tuple[float, float] = (30.0, 0.0)

        self._preprocess_polyline = TargetCentricPolyline(
            num_polylines=num_polylines,
            num_points=num_points,
            break_distance=break_distance,
            center_offset=center_offset,
        )
        self._batch_polylines = None
        self._batch_polylines_mask = None
        self._label_ids = [AgentLabel.from_str(label).value for label in labels]

        cfg = Config.from_file(model_config_path)
        is_distributed = True

        # Ego info
        self._ego_uuid = hashlib.shake_256("EGO".encode()).hexdigest(8)

        # Load Model
        self.model = build_model(cfg.model)
        self.model.eval()
        self.model.cuda()
        self.model, _ = load_checkpoint(self.model, checkpoint_path, is_distributed=is_distributed)
        self.deploy_cfg = Config.from_file(deploy_config_path)
        self.count = 0
        if build_only:
            exit(0)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._generator_uuid: RosUUID = _str_to_uuid_msg("autoware_mtr_py_")
        # publisher
        self._publisher = self.create_publisher(PredictedObjects, "~/output/objects", qos_profile)
        self._ego_traj_publisher = self.create_publisher(
            Trajectory, "~/output/trajectory", qos_profile)
        self._ego_trajectories_publisher = self.create_publisher(
            Trajectories, "~/output/trajectories", qos_profile)

    def _tracked_objects_callback(self, msg: TrackedObjects) -> None:
        timestamp = timestamp2ms(msg.header)
        states, infos = from_tracked_objects(msg)
        self._history.update(states, infos)

    def _callback(self, msg: Odometry) -> None:
        # remove invalid ancient agent history
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        current_ego, info = from_odometry(
            msg,
            uuid=self._ego_uuid,
            label_id=AgentLabel.VEHICLE,
            size=self.ego_dimensions,
        )
        self._history.update_state(current_ego, info)

        if self.count < self._num_timestamps:
            self.count = self.count + 1
            return
        pre_processed_input = {}
        # pre-process
        past_embed, polyline_info, ego_last_xyz, trajectory_mask = self._preprocess(current_ego)
        num_target, num_agent, num_time, num_feat = past_embed.shape
        pre_processed_input["obj_trajs"] = torch.Tensor(past_embed).cuda()
        pre_processed_input["obj_trajs_mask"] = trajectory_mask
        pre_processed_input["map_polylines"] = torch.Tensor(polyline_info["polylines"]).cuda()
        pre_processed_input["map_polylines_mask"] = torch.Tensor(
            polyline_info["polylines_mask"]).cuda()
        pre_processed_input["map_polylines_center"] = torch.Tensor(
            polyline_info["polyline_centers"]).cuda()
        pre_processed_input["obj_trajs_last_pos"] = torch.Tensor(
            ego_last_xyz.reshape((num_target, num_agent, 3))).cuda()
        pre_processed_input["intention_points"] = torch.Tensor(
            self._intention_points["intention_points"]).cuda()
        pre_processed_input["track_index_to_predict"] = torch.arange(
            0, num_target, dtype=torch.int32).cuda()

        # inference
        with torch.no_grad():
            pred_scores, pred_trajs = self.model(**pre_processed_input)

        # post-process
        pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)
        ego_traj = to_trajectory(header=msg.header,
                                 infos=[info],
                                 pred_scores=pred_scores,
                                 pred_trajs=pred_trajs,
                                 score_threshold=self._score_threshold,)[0]
        self._ego_traj_publisher.publish(ego_traj)

        ego_multiple_trajs = to_trajectories(header=msg.header,
                                             infos=[info],
                                             pred_scores=pred_scores,
                                             pred_trajs=pred_trajs,
                                             score_threshold=self._score_threshold, generator_uuid=self._generator_uuid)

        self._ego_trajectories_publisher.publish(ego_multiple_trajs)
        # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=[info],
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            score_threshold=self._score_threshold,
        )
        self._publisher.publish(pred_objs)

    def _postprocess(
        self,
        pred_scores: NDArray | torch.Tensor,
        pred_trajs: NDArray | torch.Tensor,
    ) -> tuple[NDArray, NDArray]:
        """Run postprocess.

        Args:
            pred_scores (NDArray | torch.Tensor): Predicted scores in the shape of
                (N, M).
            pred_trajs (NDArray | torch.Tensor): Predicted trajectories in the shape of
                (N, M, T, 7).

        Returns:
            tuple[NDArray, NDArray]: Transformed and sorted prediction.
        """
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().detach().numpy()
        if isinstance(pred_trajs, torch.Tensor):
            pred_trajs = pred_trajs.cpu().detach().numpy()
        # predicted traj point info is X,Y,Xmean,Ymean,Variance,Vx,Vy

        num_agent, num_mode, num_future, num_feat = pred_trajs.shape
        assert num_feat == 7, f"Expected predicted feature is (X, Y, Xmean, Ymean, Variance, Vx, Vy), but got {num_feat}"

        # transform from agent centric coords to world coords
        current_agent, _ = self._history.target_as_trajectory(self._ego_uuid, latest=True)

        # first 2 elements are xy then we use the negative ego rotation. reshape, I dont know
        # each trajectory is in  the ref frame of its own agent. For ego we might use TF ?
        pred_trajs[..., :2] = rotate_along_z(
            pred_trajs.reshape(num_agent, -1, num_feat)[..., :2], -current_agent.yaw
        ).reshape(num_agent, num_mode, num_future, 2)
        pred_trajs[..., :2] += current_agent.xy[:, None, None, :]

        # sort by score
        pred_scores = softmax(pred_scores, axis=1)
        sort_indices = np.argsort(-pred_scores, axis=1)
        pred_scores = np.take_along_axis(pred_scores, sort_indices, axis=1)
        pred_trajs = np.take_along_axis(pred_trajs, sort_indices[..., None, None], axis=1)

        return pred_scores, pred_trajs

    def get_embedded_inputs(self, agent_histories: List[deque[AgentState]], target_ids: List[int]):

        num_agent, num_target, num_time = int(len(agent_histories) / len(
            target_ids)), len(
            target_ids), len(agent_histories[0])
        num_type = 3

        B = num_target
        N = num_agent
        T = num_time

        past_xyz = np.ones((num_target, num_agent, num_time, 3), dtype=np.float32)
        last_xyz = np.ones((num_target, num_agent, 1, 3), dtype=np.float32)
        past_xyz_size = np.ones((num_target, num_agent, num_time, 3), dtype=np.int32)
        past_vxy = np.ones((num_target, num_agent, num_time, 2), dtype=np.float32)
        yaw_embed = np.ones((num_target, num_agent, num_time, 2), dtype=np.float32)
        timestamps = np.arange(0, num_time * 0.1, 0.1, dtype=np.float32)
        time_embed = np.zeros((num_target, num_agent, num_time, num_time + 1), dtype=np.float32)
        time_embed[:, :, np.arange(num_time), np.arange(num_time)] = 1
        time_embed[:, :, :num_time, -1] = timestamps

        type_onehot = np.zeros((num_target, num_agent, num_time, num_type + 2), dtype=np.float32)
        type_onehot[np.arange(num_target), 0, :, num_type] = 1  # Only ego is target, so index is 0
        type_onehot[:, 0, :, num_type + 1] = 1             # scenario.ego_index replaced by 0

        trajectory_mask = torch.ones(
            [num_target, num_agent, num_time], dtype=torch.bool).cuda()

        for b in range(len(target_ids)):
            for n in range(len(agent_histories)):
                history = agent_histories[b * N + n]
                for t, state in enumerate(history):
                    past_xyz[b, n, t, 0] = state.xyz[0]
                    past_xyz[b, n, t, 1] = state.xyz[1]
                    past_xyz[b, n, t, 2] = state.xyz[2]
                    last_xyz[b, n, 0, :] = state.xyz if t == T - 1 else last_xyz[b, n, 0, :]
                    label_idx = state.label_id if state.label_id != AgentLabel.UNKNOWN.value or state.label_id != AgentLabel.STATIC.value else 0
                    type_onehot[b, n, t, label_idx] = 1

                    yaw_embed[b, n, t, 0] = np.sin(state.yaw)
                    yaw_embed[b, n, t, 1] = np.cos(state.yaw)

                    past_vxy[b, n, t, 0] = state.vxy[0]
                    past_vxy[b, n, t, 1] = state.vxy[1]
                    past_xyz_size[b, n, t, 0] = state.size[0]
                    past_xyz_size[b, n, t, 1] = state.size[1]
                    past_xyz_size[b, n, t, 2] = state.size[2]
                    trajectory_mask[b, n, t] = state.is_valid

        vel_diff = np.diff(past_vxy, axis=2, prepend=past_vxy[..., 0, :][:, :, None, :])
        accel = vel_diff / 0.1
        accel[:, :, 0, :] = accel[:, :, 1, :]

        embedded_inputs = np.concatenate(
            (
                past_xyz,
                past_xyz_size,
                type_onehot,
                time_embed,
                yaw_embed,
                past_vxy,
                accel,
            ),
            axis=-1,
            dtype=np.float32,
        )
        return embedded_inputs, last_xyz, trajectory_mask

    def _preprocess(
        self,
        current_ego: AgentState,
    ):
        """Run preprocess.

        Args:
            history (AgentHistory): Ego history.
            current_ego (AgentState): Current ego state.
            lane_segments (list[LaneSegments]): Lane segments.

        Returns:

        """
        if self._batch_polylines is None or self._batch_polylines_mask is None:
            polyline_info, self._batch_polylines, self._batch_polylines_mask = self._preprocess_polyline(
                static_map=self._awml_static_map, target_state=current_ego, num_target=1, batch_polylines=None, batch_polylines_mask=None)
        else:
            polyline_info, _, __ = self._preprocess_polyline(
                static_map=self._awml_static_map, target_state=current_ego, num_target=1, batch_polylines=self._batch_polylines, batch_polylines_mask=self._batch_polylines_mask)

        sorted_histories = order_from_closest_to_furthest(
            current_ego, self._history.histories.values())
        relative_histories = get_relative_histories(
            [current_ego], sorted_histories)
        embedded_inputs, last_xyz, trajectory_mask = self.get_embedded_inputs(relative_histories, [
                                                                              0])
        return embedded_inputs, polyline_info, last_xyz, trajectory_mask


def main(args=None) -> None:
    rclpy.init(args=args)

    node = MTRNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
