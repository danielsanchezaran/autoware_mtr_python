from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np

from .agent import AgentState
from .agent import AgentTrajectory
from .agent import OriginalInfo

__all__ = ("AgentHistory",)


@dataclass
class AgentHistory:
    """A class to store agent history data."""

    max_length: int
    histories: dict[str, deque[AgentState]] = field(default_factory=dict, init=False)
    infos: dict[str, OriginalInfo] = field(default_factory=dict, init=False)

    def update(self, states: Sequence[AgentState], infos: Sequence[OriginalInfo]) -> None:
        """Update history data.

        Args:
            states (Sequence[AgentState]): Sequence of AgentStates.
        """
        for state, info in zip(states, infos, strict=True):
            self.update_state(state, info)

    def update_state(self, state: AgentState, info: OriginalInfo | None = None) -> None:
        """Update history state.

        Args:
            state (AgentState): Agent state.
        """

        uuid = state.uuid

        # Check if this UUID exists in histories; if not, initialize with unique AgentState instances
        if uuid not in self.histories.keys():
            print(f"Initializing history for UUID: {uuid}")
            self.histories[uuid] = deque(
                [AgentState(uuid=uuid) for _ in range(self.max_length)],
                maxlen=self.max_length,
            )

        # Append the new state (this automatically pops the oldest if max length is reached)
        self.histories[uuid].append(state)

        # Store additional info if provided
        if info is not None:
            self.infos[uuid] = info

    def remove_invalid(self, current_timestamp: float, threshold: float) -> None:
        """Remove agent histories whose the latest state are invalid or ancient.

        Args:
            current_timestamp (float): Current timestamp in [ms].
            threshold (float): Threshold value to filter out ancient history in [ms].
        """
        new_histories = self.histories.copy()
        new_infos = self.infos.copy()
        for uuid, history in self.histories.items():
            latest = history[-1]
            if (not latest.is_valid) or self.is_ancient(
                latest.timestamp, current_timestamp, threshold
            ):
                del new_histories[uuid]
                if uuid in new_infos:
                    del new_infos[uuid]

        self.histories = new_histories
        self.infos = new_infos

    @staticmethod
    def is_ancient(latest_timestamp: float, current_timestamp: float, threshold: float) -> bool:
        """Check whether the latest state is ancient.

        Args:
            latest_timestamp (float): Latest state timestamp in [ms].
            current_timestamp (float): Current timestamp in [ms].
            threshold (float): Timestamp threshold in [ms].

        Returns:
            bool: Return True if timestamp difference is greater than threshold,
                which means ancient.
        """
        timestamp_diff = abs(current_timestamp - latest_timestamp)
        return timestamp_diff > threshold

    def target_as_trajectory(self, target_uuid: str, latest: bool = False) -> tuple[AgentTrajectory, OriginalInfo]:
        """Convert target agent history to AgentTrajectory.

        Args:
            target_uuid (str): Target agent uuid.
            latest (bool): Whether only to return the latest trajectory,
                in the shape of (N, D). Defaults to False.

        Returns:
            tuple[AgentTrajectory, OriginalInfo]: Instanced AgentTrajectory and OriginalInfo.
        """
        if target_uuid not in self.histories:
            raise ValueError(f"Target UUID {target_uuid} not found in histories.")

        target_history = self.histories[target_uuid]
        target_info = self.infos[target_uuid]

        if latest:
            return self.get_latest_target_trajectory(target_uuid)

        waypoints = np.zeros((self.max_length, AgentTrajectory.num_dim))
        for t, state in enumerate(target_history):
            waypoints[t] = (
                *state.xyz,
                *state.size,
                state.yaw,
                *state.vxy,
                state.is_valid,
            )

        return AgentTrajectory(waypoints, target_info.label_id), target_info

    def as_trajectory(self, *, latest: bool = False) -> tuple[AgentTrajectory, list[str]]:
        """Convert agent history to AgentTrajectory.

        Args:
            latest (bool): Whether only to return the latest trajectory,
                in the shape of (N, D). Defaults to False.

        Returns:
            tuple[AgentTrajectory, list[str]]: Instanced AgentTrajectory and the list of their uuids.
        """
        if latest:
            return self._get_latest_trajectory()

        num_agent = len(self.histories)
        waypoints = np.zeros((num_agent, self.max_length, AgentTrajectory.num_dim))
        label_ids = np.zeros(num_agent, dtype=np.int64)
        uuids: list[str] = []
        for n, (uuid, history) in enumerate(self.histories.items()):
            uuids.append(uuid)
            for t, state in enumerate(history):
                waypoints[n, t] = (
                    *state.xyz,
                    *state.size,
                    state.yaw,
                    *state.vxy,
                    state.is_valid,
                )
                label_ids[n] = state.label_id

        return AgentTrajectory(waypoints, label_ids), uuids

    def get_latest_target_trajectory(self, target_uuid: str) -> tuple[AgentTrajectory, OriginalInfo]:
        """Return the latest agent state trajectory.

        Returns:
            tuple[AgentTrajectory, list[str]]: Instanced AgentTrajectory and the list of their uuids.
        """
        if target_uuid not in self.histories:
            raise ValueError(f"Target UUID {target_uuid} not found in histories.")
        num_agent = 1
        waypoints = np.zeros((num_agent, AgentTrajectory.num_dim))
        label_ids = np.zeros(num_agent, dtype=np.int64)
        uuids: list[str] = []
        state = self.histories[target_uuid][-1]
        waypoints[0] = (
            *state.xyz,
            *state.size,
            state.yaw,
            *state.vxy,
            state.is_valid,
        )
        label_ids[0] = state.label_id
        uuids.append(target_uuid)

        return AgentTrajectory(waypoints, label_ids), uuids

    def _get_latest_trajectory(self) -> tuple[AgentTrajectory, list[str]]:
        """Return the latest agent state trajectory.

        Returns:
            tuple[AgentTrajectory, list[str]]: Instanced AgentTrajectory and the list of their uuids.
        """
        num_agent = len(self.histories)
        waypoints = np.zeros((num_agent, AgentTrajectory.num_dim))
        label_ids = np.zeros(num_agent, dtype=np.int64)
        uuids: list[str] = []
        for n, (uuid, history) in enumerate(self.histories.items()):
            state = history[-1]
            waypoints[n] = (
                *state.xyz,
                *state.size,
                state.yaw,
                *state.vxy,
                state.is_valid,
            )
            label_ids[n] = state.label_id
            uuids.append(uuid)

        return AgentTrajectory(waypoints, label_ids), uuids
