from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from awml_pred.common import TRANSFORMS, load_pkl

if TYPE_CHECKING:
    from awml_pred.typing import NDArrayBool, NDArrayF32, NDArrayI32


__all__ = ("LoadIntentionPoint")


@TRANSFORMS.register()
class LoadIntentionPoint:
    """Load intention points from pkl file.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario)
        predict_all_agents (bool): Whether to predict all agents.

    Update Keys:
    ------------
        intention_points (NDArrayF32): if `only_target=True` (B, M, 2), otherwise (N, M, 2).
    """

    def __init__(self, filepath: str, target_types: list[str]) -> None:
        """Construct instance.

        Args:
        ----
            filepath (str): Pickle file path.
            only_target (bool, optional): Whether to load intention points for target agents.
                Defaults to True.

        """
        self.filepath = filepath
        self.intention_point_info = load_pkl(self.filepath)
        self.target_types = target_types

        for key in self.intention_point_info:
            print("KEY:", key)

    def __call__(self) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        intention_points = np.stack([self.intention_point_info[key]
                                    for key in self.target_types], axis=0).astype(np.float32)
        info: dict = {}
        info["intention_points"] = intention_points

        return info
