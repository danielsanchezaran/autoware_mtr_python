from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from awml_pred.common import TRANSFORMS
from awml_pred.ops import rotate_points_along_z
from autoware_mtr.dataclass.agent import AgentState
import time


if TYPE_CHECKING:
    from awml_pred.dataclass import AWMLAgentScenario, AWMLStaticMap, Trajectory
    from awml_pred.typing import NDArrayBool, NDArrayF32, NDArrayI64

__all__ = ("TargetCentricPolyline",)


@TRANSFORMS.register()
class TargetCentricPolyline:
    """Transform polylines from map coords to target centric coords.

    NOTE: current implementation returns different values compared with previous one.
        But test score is same.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
        static_map (AWMLStaticMap): `AWMLStaticMap` instance.
        predict_all_agents (bool): Whether to predict all agents.

    Updated Keys:
    -------------
        polylines (NDArrayF32): (B, K, P, D)
        polylines_mask (NDArrayBool): (B, K, P)
    """

    def __init__(
        self,
        num_polylines: int = 768,
        num_points: int = 20,
        break_distance: float = 1.0,
        center_offset: tuple[float, float] = (30.0, 0.0),
    ) -> None:
        """Construct instance.

        Args:
        ----
            num_polylines (int, optional): Max number of polylines can be contained. Defaults to 768.
            num_points (int, optional): Max number of points, which each polyline can contain. Defaults to 20.
            break_distance (float, optional): The distance threshold to separate polyline into two polylines.
                Defaults to 1.0.
            center_offset (tuple[float, float], optional): The offset position. Defaults to (30.0, 0.0).

        """
        self.num_polylines = num_polylines
        self.num_points = num_points
        self.break_distance = break_distance
        self.center_offset = center_offset

    def _do_transform(
        self,
        polylines: NDArrayF32,
        polylines_mask: NDArrayBool,
        current_target: AgentState,
        num_target: int,
    ) -> tuple[NDArrayF32, NDArrayBool]:
        """Transform polylines from map coords to target centric coords.

        Args:
        ----
            polylines (NDArrayF32): in shape (K, P, Dp).
            polylines_mask (NDArrayBool): in shape (K, P).
            current_target (Trajectory): in shape (B, Da).

        Returns:
        -------
            tuple[NDArrayF32, NDArrayBool]: Transformed results.
                `polylines`: in shape (B, K, P, Dp).
                `polylines_mask`: in shape (B, K, P).

        """

        polylines[..., :3] -= current_target.xyz
        polylines[..., :2] = rotate_points_along_z(
            points=polylines[..., 0:2].reshape(num_target, -1, 2),
            angle=-current_target.yaw,
        ).reshape(num_target, -1, self.num_points, 2)
        polylines[..., 3:5] = rotate_points_along_z(
            points=polylines[..., 3:5].reshape(num_target, -1, 2),
            angle=-current_target.yaw,
        ).reshape(num_target, -1, self.num_points, 2)

        xy_pos_pre = polylines[..., 0:2]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        polylines = np.concatenate((polylines, xy_pos_pre), axis=-1)
        polylines[~polylines_mask] = 0
        return polylines, polylines_mask

    @staticmethod
    def _load_polyline_center(polyline: NDArrayF32, polyline_mask: NDArrayBool) -> NDArrayF32:
        tmp_sum = (polyline[..., :3] * polyline_mask[..., None]).sum(axis=-2, dtype=np.float32)
        polyline_center: NDArrayF32 = tmp_sum / np.clip(
            polyline_mask.sum(axis=-1, dtype=np.float32)[..., None],
            a_min=1.0,
            a_max=None,
        )
        return polyline_center.astype(np.float32)

    def _generate_batch(self, polylines: NDArrayF32) -> tuple[NDArrayF32, NDArrayBool]:
        """Generate batch polylines from points shape with (N, Dp) to (K, P, Dp).

        Args:
        ----
            polylines (NDArrayF32): Points, in shape (N, D).

        Returns:
        -------
            tuple[NDArrayF32, NDArrayBool]: Separated polylines and its mask.
                `ret_polylines`: Batch polylines, in shape (K, P, Dp).
                `ret_polylines_mask`: Mask of polylines, in shape (K, P).

        """
        point_dim = polylines.shape[-1]
        polyline_shifts = np.roll(polylines, shift=1, axis=0)
        buffer = np.concatenate((polylines[:, 0:2], polyline_shifts[:, 0:2]), axis=-1)
        buffer[0, 2:4] = buffer[0, 0:2]

        break_idxs: NDArrayI64 = (
            np.linalg.norm(buffer[:, 0:2] - buffer[:, 2:4], axis=-1) > self.break_distance
        ).nonzero()[0]
        polyline_list: list[NDArrayF32] = np.array_split(polylines, break_idxs, axis=0)

        ret_polylines, ret_polylines_mask = [], []

        def append_single_polyline(new_polyline: NDArrayF32) -> None:
            num_new_polyline = len(new_polyline)
            cur_polyline = np.zeros((self.num_points, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((self.num_points), dtype=np.int32)
            cur_polyline[:num_new_polyline] = new_polyline
            cur_valid_mask[:num_new_polyline] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for line in polyline_list:
            num_pts = len(line)
            if num_pts <= 0:
                continue
            for idx in range(0, num_pts, self.num_points):
                append_single_polyline(line[idx: idx + self.num_points])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0) > 0

        return ret_polylines, ret_polylines_mask

    def __call__(self, static_map: AWMLStaticMap, target_state: AgentState, num_target: int,  batch_polylines=None, batch_polylines_mask=None) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """

        if batch_polylines is None or batch_polylines_mask is None:
            all_polylines: NDArrayF32 = static_map.get_all_polyline(as_array=True, full=True)
            batch_polylines, batch_polylines_mask = self._generate_batch(all_polylines)

        ret_polylines: NDArrayF32
        ret_polylines_mask: NDArrayBool
        if len(batch_polylines) > self.num_polylines:
            polyline_center: NDArrayF32 = batch_polylines[..., :2].sum(axis=1) / np.clip(
                batch_polylines_mask.sum(axis=-1, dtype=np.float32)[:, None],
                a_min=1.0,
                a_max=None,
            )
            center_offset: NDArrayF32 = np.array(self.center_offset, dtype=np.float32)[None, :].repeat(
                num_target,
                axis=0,
            )
            center_offset = rotate_points_along_z(
                points=center_offset.reshape(num_target, 1, 2),
                angle=target_state.yaw,
            ).reshape(num_target, 2)

            center_pos = target_state.xy + center_offset
            distances: NDArrayF32 = np.linalg.norm(
                center_pos[:, None, :] - polyline_center[None, ...], axis=-1)
            topk_idxs = np.argsort(distances, axis=1)[:, : self.num_polylines]
            ret_polylines = batch_polylines[topk_idxs]
            ret_polylines_mask = batch_polylines_mask[topk_idxs]
        else:
            ret_polylines = batch_polylines[None, ...].repeat(num_target, axis=0)
            ret_polylines_mask = batch_polylines_mask[None, ...].repeat(num_target, axis=0)
        ret_polylines, ret_polylines_mask = self._do_transform(
            ret_polylines, ret_polylines_mask, target_state, num_target)
        info: dict = {}
        info["polylines"] = ret_polylines
        info["polylines_mask"] = ret_polylines_mask > 0
        info["polyline_centers"] = self._load_polyline_center(ret_polylines, ret_polylines_mask)

        return info, batch_polylines, batch_polylines_mask
