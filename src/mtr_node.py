import hashlib
import torch
import numpy as np
from collections import deque
from copy import deepcopy
import numpy as np
import math
import time

from scipy.interpolate import interp1d
from typing import List

import rclpy
from rclpy.duration import Duration

import rclpy.parameter
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import Parameter
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from nav_msgs.msg import Odometry
from std_msgs.msg import String


from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor
from utils.polyline import TargetCentricPolyline

from autoware_perception_msgs.msg import PredictedObjects
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_new_planning_msgs.msg import Trajectories, TrajectoryGeneratorInfo
from autoware_new_planning_msgs.msg import Trajectory as NewTrajectory
from unique_identifier_msgs.msg import UUID as RosUUID
from autoware_mtr.dataclass.agent import _str_to_uuid_msg, OriginalInfo
from geometry_msgs.msg import Point

from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjects

from awml_pred.common import Config, load_checkpoint
from awml_pred.models import build_model
from utils.lanelet_converter import convert_lanelet
from utils.constant import MAP_TYPE_COLORS
from utils.load import LoadIntentionPoint
from autoware_mtr.conversion.ego import from_odometry, from_trajectory_point
from autoware_mtr.conversion.tracked_object import from_tracked_objects
from autoware_mtr.conversion.misc import timestamp2ms, yaw_from_quaternion
from autoware_mtr.conversion.trajectory import get_relative_histories, order_from_closest_to_furthest, to_trajectories, _yaw_to_quaternion
from autoware_mtr.dataclass.static_map import AWMLStaticMap
from autoware_mtr.datatype import AgentLabel
from autoware_mtr.geometry import rotate_along_z
from autoware_mtr.dataclass.history import AgentHistory
from autoware_mtr.dataclass.agent import AgentState, AgentTrajectory
from autoware_mtr.conversion.predicted_object import to_predicted_objects
from typing import List
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


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

        self._prev_trajectory_sub = self.create_subscription(
            Trajectory, "/planning/scenario_planning/trajectory", self._previous_trajectory_callback, qos_profile_2)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._odometry_subscription = self.create_subscription(
            Odometry,
            "~/input/ego",
            self._odometry_callback,
            qos_profile,
        )

        self._timer = self.create_timer(0.1, self._callback)

        # ROS parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)

        model_config_path = (
            self.declare_parameter("model_config", descriptor=descriptor)
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

        self._publish_debug_polyline_batch = (
            self.declare_parameter("publish_debug_polyline_batch", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        self._publish_debug_polyline_map = (
            self.declare_parameter("publish_debug_polyline_map", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
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

        self.propagate_future_states = (self.declare_parameter(
            "propagate_future_states", False, ParameterDescriptor(
                description='Propagate future states',
                type=Parameter.Type.BOOL.value
            )).get_parameter_value().bool_value)

        self.add_left_bias_history = (self.declare_parameter(
            "add_left_bias_history", False, ParameterDescriptor(
                description='Propagate future states',
                type=Parameter.Type.BOOL.value
            )).get_parameter_value().bool_value)

        self.add_right_bias_history = (self.declare_parameter(
            "add_right_bias_history", False, ParameterDescriptor(
                description='Propagate future states',
                type=Parameter.Type.BOOL.value
            )).get_parameter_value().bool_value)

        self.future_state_propagation_sec = (self.declare_parameter(
            "future_state_propagation_sec", descriptor=descriptor).get_parameter_value().double_value)

        self._num_timestamps = num_timestamp
        self._history = AgentHistory(max_length=num_timestamp)
        self._future_propagated_history = AgentHistory(max_length=num_timestamp)
        self._awml_static_map: AWMLStaticMap = convert_lanelet(lanelet_file)
        self.current_ego, self.current_ego_info = None, None

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
        self._polyline_center = None
        self._prev_trajectory: Trajectory | None = None
        self._last_ego: AgentState | None = None
        self._label_ids = [AgentLabel.from_str(label).value for label in labels]

        cfg = Config.from_file(model_config_path)
        is_distributed = True

        # Ego info
        self._ego_uuid = hashlib.shake_256("EGO".encode()).hexdigest(8)
        self._ego_uuid_future = hashlib.shake_256("EGO_FUTURE".encode()).hexdigest(8)

        # Load Model
        self.model = build_model(cfg.model)
        self.model.eval()
        self.model.cuda()
        self.model, _ = load_checkpoint(self.model, checkpoint_path, is_distributed=is_distributed)
        self.count = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._min_prediction_time = 7.0

        self._generator_uuid: RosUUID = _str_to_uuid_msg("autoware_mtr_py_")
        # publisher
        self._publisher = self.create_publisher(PredictedObjects, "~/output/objects", qos_profile)
        self._ego_trajectories_publisher = self.create_publisher(
            Trajectories, "~/output/trajectories", qos_profile)

        self._debug_polylines_pub = self.create_publisher(
            MarkerArray,
            "/debug_polylines",
            1,
        )

        # Add a callback for parameter changes
        self.add_on_set_parameters_callback(self._parameter_callback)

    def _pub_debug_polylines(self, polylines: NDArray, polylines_mask: NDArray, header: Header, ego_state: AgentState | None = None, polyline_centers: NDArray | None = None):
        marker_array = MarkerArray()
        _, num_polylines, num_points, point_dim = polylines.shape if len(
            polylines.shape) == 4 else (1, polylines.shape[0], polylines.shape[1], polylines
                                        .shape[2])
        for i in range(num_polylines):
            marker = Marker()
            marker.type = marker.LINE_STRIP
            marker.header.stamp = header.stamp
            marker.header.frame_id = header.frame_id
            marker.id = i
            marker.action = marker.ADD

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.0

            marker.lifetime.nanosec = 100000000
            marker.frame_locked = True

            marker.points = []
            polyline = polylines[0][i] if len(polylines.shape) == 4 else polylines[i]
            mask = polylines_mask[0][i]
            for p, m in zip(polyline, mask):
                if not m:
                    continue
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = MAP_TYPE_COLORS[int(
                    p[-3])] if point_dim == 9 else MAP_TYPE_COLORS[int(
                        p[-1])]
                tmp_point = Point()
                tmp_point.x = float(p[0])
                tmp_point.y = float(p[1])
                tmp_point.z = 0.2
                marker.points.append(tmp_point)

            marker_array.markers.append(marker)

        if (polyline_centers is not None):
            _, num_polylines, center_dim = polyline_centers.shape
            for i in range(num_polylines):
                marker = Marker()
                marker.type = marker.SPHERE
                marker.header.stamp = header.stamp
                marker.header.frame_id = header.frame_id
                marker.id = i + num_polylines
                marker.action = marker.ADD

                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                marker.color.a = 0.99
                marker.color.r = 0.9
                marker.color.g = 0.1
                marker.color.b = 0.1

                marker.lifetime.nanosec = 100000000
                marker.frame_locked = True

                marker.pose.position.x = float(polyline_centers[0][i][0])
                marker.pose.position.y = float(polyline_centers[0][i][1])
                marker.pose.position.z = 0.0

                marker_array.markers.append(marker)

        self._debug_polylines_pub.publish(marker_array)

    def _parameter_callback(self, params):
        for param in params:
            if param.name == "propagate_future_states":
                self.propagate_future_states = param.value

        # Return success
        return rclpy.parameter.ParameterValue(
            successful=True
        )

    def _create_pre_processed_input(self, current_ego: AgentState, history: AgentHistory):
        past_embed, polyline_info, ego_last_xyz, trajectory_mask = self._preprocess(
            current_ego, history)
        num_target, num_agent, num_time, num_feat = past_embed.shape
        pre_processed_input = {}
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
        return pre_processed_input

    def _do_predictions(self, true_ego_state: AgentState, ego_states: List[AgentState], infos: List[OriginalInfo], histories: List[AgentHistory], requires_concatenation: List[bool], uuids: List[RosUUID]):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        out_objects: PredictedObjects = PredictedObjects()
        out_objects.header = header

        out_trajectories: Trajectories = Trajectories()
        generator_name: String = String()
        generator_name.data = "mtr"
        out_trajectories.generator_info = [TrajectoryGeneratorInfo(
            generator_id=self._generator_uuid, generator_name=generator_name)]

        for ego_state, info, history, concatenate, uuid in zip(ego_states, infos, histories, requires_concatenation, uuids):
            pre_processed_input = self._create_pre_processed_input(ego_state, history)
            # inference
            with torch.no_grad():
                pred_scores, pred_trajs = self.model(**pre_processed_input)

            # post-process
            current_target_trajectory, _ = history.target_as_trajectory(
                uuid, latest=True)
            pred_scores, pred_trajs = self._postprocess(
                pred_scores, pred_trajs, current_target_trajectory)

            ego_multiple_trajs = to_trajectories(header=header,
                                                 infos=[info],
                                                 pred_scores=pred_scores,
                                                 pred_trajs=pred_trajs,
                                                 score_threshold=self._score_threshold, generator_uuid=self._generator_uuid)

            if concatenate:
                ego_multiple_trajs = self.simple_trajectory_concatenation(
                    self._prev_trajectory, ego_multiple_trajs, true_ego_state)

            # convert to ROS msg
            pred_objs = to_predicted_objects(
                header=header,
                infos=[info],
                pred_scores=pred_scores,
                pred_trajs=pred_trajs,
                score_threshold=self._score_threshold,
            )

            for trajectory in ego_multiple_trajs.trajectories:
                out_trajectories.trajectories.append(trajectory)

            for predicted_object in pred_objs.objects:
                out_objects.objects.append(predicted_object)

            if (self._publish_debug_polyline_map):
                if np.linalg.norm(ego_state.xy - true_ego_state.xy) > 1e-3:
                    continue
                header_map = Header()
                header_map .stamp = self.get_clock().now().to_msg()
                header_map .frame_id = "base_link"
                self._pub_debug_polylines(pre_processed_input["map_polylines"].cpu().detach().numpy(),
                                          pre_processed_input["map_polylines_mask"].cpu().detach().numpy(), header_map, None, pre_processed_input["map_polylines_center"].cpu().detach().numpy())
        return out_trajectories, out_objects

    def _generate_steering_bias(self, base_agent_state: AgentState, base_agent_info: OriginalInfo, base_agent_history: AgentHistory, yaw_bias: float, bias_left: bool = True, bias_right: bool = True):
        def normalize_angle(angle: float) -> float:
            """Normalize angle to range [-π, π]."""
            return (angle + math.pi) % (2 * math.pi) - math.pi

        def move_history_and_info(histories: AgentHistory, old_key: str, new_key: str):
            if old_key in histories.histories and old_key in histories.infos:
                histories.histories[new_key] = histories.histories.pop(old_key)
                histories.infos[new_key] = histories.infos.pop(old_key)

        def get_modified_agent_info(original_uuid: str, uuid: str, base_agent_state: AgentState, base_agent_info: OriginalInfo, base_agent_history: AgentHistory, new_yaw: float):
            biased_state: AgentState = AgentState(uuid=uuid,
                                                  timestamp=base_agent_state.timestamp,
                                                  label_id=base_agent_state.label_id,
                                                  xyz=base_agent_state.xyz,
                                                  size=base_agent_state.size,
                                                  yaw=new_yaw,
                                                  vxy=base_agent_state.vxy,
                                                  is_valid=base_agent_state.is_valid)

            new_kinematics = base_agent_info.kinematics
            new_kinematics.pose_with_covariance.pose.orientation = _yaw_to_quaternion(
                new_yaw)

            ros_uuid = _str_to_uuid_msg(uuid)
            biased_info: OriginalInfo = OriginalInfo(uuid=ros_uuid,
                                                     kinematics=new_kinematics,
                                                     classification=base_agent_info.classification,
                                                     shape=base_agent_info.shape,
                                                     existence_probability=base_agent_info.existence_probability)

            biased_history: AgentHistory = deepcopy(base_agent_history)
            move_history_and_info(biased_history, original_uuid, uuid)
            biased_history.update_state(biased_state, biased_info)
            return biased_state, biased_info, biased_history

        original_yaw = base_agent_state.yaw
        yaws = []
        uuids = []
        biased_states = []
        biased_infos = []
        biased_histories = []

        if bias_left:
            yaws.append(normalize_angle(original_yaw + abs(yaw_bias)))
            uuids.append(hashlib.shake_256("EGO_LEFT".encode()).hexdigest(8))
        if bias_right:
            yaws.append(normalize_angle(original_yaw - abs(yaw_bias)))
            uuids.append(hashlib.shake_256("EGO_RIGHT".encode()).hexdigest(8))

        for yaw, uuid in zip(yaws, uuids):
            biased_state, biased_info, biased_history = get_modified_agent_info(original_uuid=self._ego_uuid,
                                                                                uuid=uuid, base_agent_state=base_agent_state, base_agent_info=base_agent_info, base_agent_history=base_agent_history, new_yaw=yaw)
            biased_states.append(biased_state)
            biased_infos.append(biased_info)
            biased_histories.append(biased_history)
        return biased_states, biased_infos, biased_histories, uuids

    def _previous_trajectory_callback(self, msg: Trajectory) -> None:
        self._prev_trajectory = msg

    def _tracked_objects_callback(self, msg: TrackedObjects) -> None:
        timestamp = timestamp2ms(msg.header)
        states, infos = from_tracked_objects(msg)
        self._history.update(states, infos)

    def _odometry_callback(self, msg: Odometry) -> None:
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        self.current_ego, self.current_ego_info = from_odometry(
            msg,
            uuid=self._ego_uuid,
            label_id=AgentLabel.VEHICLE,
            size=self.ego_dimensions,
        )

    def _callback(self) -> None:
        # remove invalid ancient agent history
        if self.current_ego is None or self.current_ego_info is None:
            return

        self._history.update_state(self.current_ego, self.current_ego_info)
        if self.count < self._num_timestamps:
            self.count = self.count + 1
            return

        true_ego_state = deepcopy(self.current_ego)
        ego_states = [self.current_ego]
        infos = [self.current_ego_info]
        histories = [self._history]
        requires_concatenation = [False]
        uuids = [self._ego_uuid]

        propagation_required = self.propagate_future_states or self.add_left_bias_history or self.add_right_bias_history

        if propagation_required and self._prev_trajectory is not None and len(self._prev_trajectory.points) > 2:
            history_from_traj, future_ego_state, future_ego_info = self.get_ego_history_from_trajectory(
                self._prev_trajectory, self.future_state_propagation_sec)
            if history_from_traj is not None and self.propagate_future_states:
                ego_states.append(future_ego_state)
                infos.append(future_ego_info)
                histories.append(history_from_traj)
                requires_concatenation.append(True)
                uuids.append(self._ego_uuid_future)

            if self.add_left_bias_history or self.add_right_bias_history:
                biased_states, biased_infos, biased_histories, bias_uuids = self._generate_steering_bias(
                    future_ego_state, future_ego_info, history_from_traj, math.pi/18, bias_left=self.add_left_bias_history, bias_right=self.add_right_bias_history)
                for biased_state, biased_info, biased_history, bias_uuid in zip(biased_states, biased_infos, biased_histories, bias_uuids):
                    ego_states.append(biased_state)
                    infos.append(biased_info)
                    histories.append(biased_history)
                    requires_concatenation.append(True)
                    uuids.append(bias_uuid)

        ego_multiple_trajs, pred_objs = self._do_predictions(
            true_ego_state=true_ego_state, ego_states=ego_states, infos=infos, histories=histories, requires_concatenation=requires_concatenation, uuids=uuids)

        self._ego_trajectories_publisher.publish(ego_multiple_trajs)
        self._publisher.publish(pred_objs)
        if self._publish_debug_polyline_batch:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            self._pub_debug_polylines(self._batch_polylines,
                                      self._batch_polylines_mask, header)

    def _postprocess(
        self,
        pred_scores: NDArray | torch.Tensor,
        pred_trajs: NDArray | torch.Tensor,
        current_agent: AgentTrajectory,
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

        past_xyz = np.ones((B, N, T, 3), dtype=np.float32)
        last_xyz = np.ones((B, N, 1, 3), dtype=np.float32)
        past_xyz_size = np.ones((B, N, T, 3), dtype=np.int32)
        past_vxy = np.ones((B, N, T, 2), dtype=np.float32)
        yaw_embed = np.ones((B, N, T, 2), dtype=np.float32)
        timestamps = np.arange(0, T * 0.1, 0.1, dtype=np.float32)
        time_embed = np.zeros((B, N, T, T + 1), dtype=np.float32)
        time_embed[:, :, np.arange(T), np.arange(T)] = 1
        time_embed[:, :, :T, -1] = timestamps

        type_onehot = np.zeros((B, N, T, num_type + 2), dtype=np.float32)
        type_onehot[np.arange(B), 0, :, num_type] = 1  # Only ego is target, so index is 0
        type_onehot[:, 0, :, num_type + 1] = 1             # scenario.ego_index replaced by 0

        trajectory_mask = torch.ones(
            [B, N, T], dtype=torch.bool).cuda()

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
        history: AgentHistory
    ):
        """Run preprocess.

        Args:
            history (AgentHistory): Ego history.
            current_ego (AgentState): Current ego state.
            lane_segments (list[LaneSegments]): Lane segments.

        Returns:

        """
        polyline_info, self._batch_polylines, self._batch_polylines_mask, self._polyline_center = self._preprocess_polyline(
            static_map=self._awml_static_map, target_state=current_ego, num_target=1, batch_polylines=self._batch_polylines, batch_polylines_mask=self._batch_polylines_mask, polyline_center=self._polyline_center)

        sorted_histories = order_from_closest_to_furthest(
            current_ego, history.histories.values())
        relative_histories = get_relative_histories(
            [current_ego], sorted_histories)
        embedded_inputs, last_xyz, trajectory_mask = self.get_embedded_inputs(relative_histories, [
                                                                              0])
        return embedded_inputs, polyline_info, last_xyz, trajectory_mask

    def interpolate_trajectory(self, original_traj: Trajectory, start_time: float) -> Trajectory:
        """
        Interpolates a segment of the given trajectory from start_time, generating points at 0.1s intervals for 1 second.

        Args:
            original_traj (Trajectory): The original trajectory to interpolate.
            start_time (float): The time to start interpolation from.

        Returns:
            Trajectory: A new trajectory with interpolated points.
        """
        if not original_traj.points:
            return Trajectory()

        # Extract time, positions, velocities, and yaw
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec *
                         1e-9 for p in original_traj.points])
        x = np.array([p.pose.position.x for p in original_traj.points])
        y = np.array([p.pose.position.y for p in original_traj.points])
        vx = np.array([p.longitudinal_velocity_mps for p in original_traj.points])
        vy = np.array([p.lateral_velocity_mps for p in original_traj.points])
        yaw = np.array([yaw_from_quaternion(p.pose.orientation)
                       for p in original_traj.points])  # Extract yaw

        # Find the closest index BEFORE start_time
        idx_start = np.searchsorted(times, start_time) - 1
        idx_start = max(0, idx_start)  # Ensure valid index

        # Define new time samples every 0.1s for 1 second interval
        t_interp = np.arange(start_time, start_time + 1.0 + 1e-6, 0.1)

        # Ensure valid interpolation range
        if times[-1] < t_interp[-1]:  # Not enough data to interpolate fully
            return original_traj
            # raise ValueError("Not enough trajectory data to interpolate full 1-second interval.")

        # Create interpolators
        interp_x = interp1d(times, x, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(times, y, kind='linear', fill_value="extrapolate")
        interp_vx = interp1d(times, vx, kind='linear', fill_value="extrapolate")
        interp_vy = interp1d(times, vy, kind='linear', fill_value="extrapolate")
        interp_yaw = interp1d(times, yaw, kind='linear',
                              fill_value="extrapolate")  # Interpolate yaw!

        # Generate interpolated values
        x_interp = interp_x(t_interp)
        y_interp = interp_y(t_interp)
        vx_interp = interp_vx(t_interp)
        vy_interp = interp_vy(t_interp)
        yaw_interp = interp_yaw(t_interp)  # Get smoothed yaw directly

        # Create a new Trajectory message
        new_traj = Trajectory()

        for i, t in enumerate(t_interp):
            traj_point = TrajectoryPoint()
            traj_point.pose.position.x = float(x_interp[i])
            traj_point.pose.position.y = float(y_interp[i])
            # Keep Z constant
            traj_point.pose.position.z = original_traj.points[idx_start].pose.position.z
            traj_point.longitudinal_velocity_mps = float(vx_interp[i])
            traj_point.lateral_velocity_mps = float(vy_interp[i])
            traj_point.time_from_start = Duration(
                seconds=int(t), nanoseconds=int((t % 1) * 1e9)).to_msg()

            # Use interpolated yaw instead of recomputing
            traj_point.pose.orientation = _yaw_to_quaternion(yaw_interp[i])

            new_traj.points.append(traj_point)

        return new_traj

    def get_ego_history_from_trajectory(self, previous_best_trajectory: Trajectory, time_start: float):
        if (previous_best_trajectory is None or len(previous_best_trajectory.points) == 0):
            return None, None, None

        # last_point: TrajectoryPoint = previous_best_trajectory.points[-1]

        # time_start: float = self.get_time_float(
        #     last_point.time_from_start) - time_from_end_of_trajectory
        time_start = max(0.0, time_start)

        interpolated_trajectory: NewTrajectory = self.interpolate_trajectory(
            previous_best_trajectory, time_start)
        history = AgentHistory(max_length=self._num_timestamps)
        start_index = 0
        end_index = len(interpolated_trajectory.points) - 1
        for i in range(start_index, end_index):
            point = interpolated_trajectory.points[i]
            state, info = from_trajectory_point(point=point, uuid=self._ego_uuid_future, header=previous_best_trajectory.header, label_id=AgentLabel.VEHICLE,
                                                size=self.ego_dimensions)
            history.update_state(state, info)
        return history, state, info

    def find_nearest_index_point(self, trajectory: NewTrajectory, point: TrajectoryPoint):
        x, y = point.pose.position.x, point.pose.position.y
        min_distance = 100000.0
        nearest_index = 0

        for i, point in enumerate(trajectory.points):
            dist = (point.pose.position.x - x) ** 2 + (point.pose.position.y - y)
            if dist < min_distance:
                min_distance = dist
                nearest_index = i
        return nearest_index

    def find_nearest_index(self, trajectory: NewTrajectory, current_ego: AgentState):
        x, y = current_ego.xy
        min_distance = 100000.0
        nearest_index = 0

        for i, point in enumerate(trajectory.points):
            dist = (point.pose.position.x - x) ** 2 + (point.pose.position.y - y)
            if dist < min_distance:
                min_distance = dist
                nearest_index = i
        return nearest_index

    def simple_trajectory_concatenation(self, base_trajectory: Trajectory, trajectories: Trajectories, ego_state: AgentState) -> Trajectories:
        output = Trajectories()
        output.generator_info = trajectories.generator_info
        base_size = len(base_trajectory.points)

        closest_ego_index = self.find_nearest_index(base_trajectory, ego_state)

        for i in range(len(trajectories.trajectories)):
            trajectory = trajectories.trajectories[i]
            closest_point_between_trajectories = self.find_nearest_index_point(
                base_trajectory, trajectory.points[0]) - 1
            closest_point_between_trajectories = max(
                max(closest_point_between_trajectories, 0), closest_ego_index + 1)
            added_traj_length = len(trajectory.points)
            new_trajectory = NewTrajectory()
            new_trajectory.header = trajectory.header
            new_trajectory.generator_id = self._generator_uuid
            new_trajectory.points = base_trajectory.points[:closest_point_between_trajectories]

            last_time_from_previous_path = max(self.get_time_float(new_trajectory.points[-1].time_from_start) - self.get_time_float(
                new_trajectory.points[closest_ego_index].time_from_start), 0.0)
            last_point_time = last_time_from_previous_path
            j = 0
            while j < len(trajectory.points) and last_point_time < self._min_prediction_time * 2.0:
                last_point_time = last_time_from_previous_path + \
                    self.get_time_float(trajectory.points[j].time_from_start)
                new_trajectory.points.append(trajectory.points[j])
                j += 1
            output.trajectories.append(new_trajectory)
        return output

    def to_new_trajectory(self, trajectory: Trajectory, generator_id) -> NewTrajectory:
        output = NewTrajectory()
        output.points = trajectory.points
        output.header = trajectory.header
        output.header.stamp = self.get_clock().now().to_msg()
        output.generator_id = generator_id
        return output

    def concatenate_trajectories(self, base_trajectory: Trajectory, trajectories: Trajectories, ego_state: AgentState) -> Trajectories:
        base_trajectory_new_format = self.to_new_trajectory(
            self.interpolate_trajectory(base_trajectory, 0.0), self._generator_uuid)
        ego_index = self.find_nearest_index(base_trajectory_new_format, ego_state)
        cropped_base_trajectory = self.crop_trajectory(
            base_trajectory_new_format, ego_index)
        cropped_base_trajectory.header.stamp = self.get_clock().now().to_msg()

        output = deepcopy(trajectories)
        output.trajectories = []
        for trajectory in trajectories.trajectories:
            out_trajectory = self.concatenate_trajectory(
                cropped_base_trajectory, trajectory, ego_index)
            output.trajectories.append(out_trajectory)
        return output

    def get_time_float(self, duration: Duration) -> float:
        return duration.sec + float(duration.nanosec) * 1e-9


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
