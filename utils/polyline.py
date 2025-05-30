from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from awml_pred.common import TRANSFORMS
from autoware_mtr.geometry import rotate_along_z
from autoware_mtr.dataclass.agent import AgentState
import time
from numba import njit, prange


if TYPE_CHECKING:
    from autoware_mtr.dataclass.static_map import AWMLStaticMap
    from awml_pred.typing import NDArrayBool, NDArrayF32, NDArrayI64

__all__ = ("TargetCentricPolyline",)


@njit(parallel=True)
def compute_polyline_centers_batch(polylines, masks):
    batch_size, num_polylines, num_points, dim = polylines.shape
    centers = np.empty((batch_size, num_polylines, 3), dtype=np.float32)

    for b in prange(batch_size):
        for i in prange(num_polylines):
            polyline = polylines[b, i, :, :3]
            mask = masks[b, i]
            valid_points = polyline[mask, :3]

            if len(valid_points) < 2:
                if len(valid_points) > 0:
                    centers[b, i] = valid_points[0, :3]
                else:
                    centers[b, i] = np.array([np.nan, np.nan], dtype=np.float32)
                continue

            diffs = valid_points[1:] - valid_points[:-1]  # (N-1, 3)
            segment_lengths = np.empty(len(diffs), dtype=np.float32)
            for j in range(len(diffs)):
                segment_lengths[j] = np.sqrt(diffs[j, 0]**2 + diffs[j, 1]**2 + diffs[j, 2]**2)
            cumulative_length = np.empty(len(segment_lengths) + 1)
            cumulative_length[0] = 0
            for j in range(len(segment_lengths)):
                cumulative_length[j + 1] = cumulative_length[j] + segment_lengths[j]

            total_length = cumulative_length[-1]
            mid_length = total_length / 2

            idx = np.searchsorted(cumulative_length, mid_length) - 1
            if idx < 0:
                idx = 0
            den = cumulative_length[idx + 1] - cumulative_length[idx]
            t = (mid_length - cumulative_length[idx]) / den if abs(den) > 1e-6 else 0
            centers[b, i] = (1 - t) * valid_points[idx, :3] + t * valid_points[idx + 1, :3]

    return centers


@TRANSFORMS.register()
class TargetCentricPolyline:
    """Transform polylines from map coords to target centric coords.

    NOTE: current implementation returns different values compared with previous one.
        But test score is same.

    Required Keys:
    --------------
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
        polylines[..., :2] = rotate_along_z(
            points=polylines[..., 0:2].reshape(num_target, -1, 2),
            angle=-current_target.yaw,
        ).reshape(num_target, -1, self.num_points, 2)
        polylines[..., 3:5] = rotate_along_z(
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
    def _load_polyline_center(polyline, mask):
        """
        Finds the center of arc length for a polyline while considering only valid points.

        Args:
            polyline (np.ndarray): A (20, 7) array representing a polyline.
            mask (np.ndarray): A (20,) boolean array indicating valid points.

        Returns:
            np.ndarray: The interpolated (x, y) coordinate of the arc-length center.
        """
        # Extract valid points
        valid_points = polyline[mask, :3]  # Shape (N_valid, 3) where N_valid ≤ 20

        if len(valid_points) < 2:
            # If less than 2 valid points, return the first valid point or NaN if none
            return valid_points[0] if len(valid_points) > 0 else np.array([np.nan, np.nan])

        # Compute segment distances
        segment_lengths = np.linalg.norm(np.diff(valid_points, axis=0), axis=1)

        # Compute cumulative arc length
        cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0)

        # Find the total length and midpoint
        total_length = cumulative_length[-1]
        mid_length = total_length / 2

        # Find the segment where the midpoint length falls
        idx = np.searchsorted(cumulative_length, mid_length) - 1

        # Linear interpolation
        den = cumulative_length[idx + 1] - cumulative_length[idx]
        t = (mid_length - cumulative_length[idx]) / den if abs(den) > 1e-6 else 0
        center_point = (1 - t) * valid_points[idx] + t * valid_points[idx + 1]

        return center_point

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

    def __call__(self, static_map: AWMLStaticMap, target_state: AgentState, num_target: int,  batch_polylines=None, batch_polylines_mask=None, polyline_center: NDArrayF32 | None = None) -> dict:
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
            if polyline_center is None:
                polyline_center: NDArrayF32 = np.array([
                    self._load_polyline_center(polyline, mask)
                    for polyline, mask in zip(batch_polylines, batch_polylines_mask)
                ])[..., :2]

            center_offset: NDArrayF32 = np.array(self.center_offset, dtype=np.float32)[None, :].repeat(
                num_target,
                axis=0,
            )
            center_offset = rotate_along_z(
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

        info["polyline_centers"] = compute_polyline_centers_batch(
            ret_polylines, ret_polylines_mask)
        return info, batch_polylines, batch_polylines_mask, polyline_center
