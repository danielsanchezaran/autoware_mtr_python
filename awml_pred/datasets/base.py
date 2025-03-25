from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from awml_pred.common import load_pkl
from awml_pred.dataclass import AWMLAgentScenario, AWMLStaticMap
from awml_pred.datatype import DatasetName

from .transforms import Compose

if TYPE_CHECKING:
    from awml_pred.dataclass import EvaluationData
    from awml_pred.typing import Tensor

__all__ = ("BaseDataset",)


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Abstract base class for Dataset."""

    VALID_OBJECT_TYPES: list[str]

    def __init__(
        self,
        data_root: str,
        scenario_dir: str,
        ann_file: str,
        agent_types: list[str],
        transforms: Sequence[dict] | None = None,
        *,
        predict_all_agents: bool = False,
        training: bool = True,
    ) -> None:
        """Construct instance.

        Args:
        ----
            data_root (str): Root directory path of data.
            scenario_dir (str): Directory name existing scenario infos (.pkl).
            ann_file (str): Annotation file (.pkl) path.
            agent_types (list[str]): List of agent type names to be considered in this experiment.
            transforms (Sequence[dict] | None, optional): List of transform information.
            predict_all_agents (bool, optional): Whether to predict all agents. Defaults to False.
            training (bool, optional): Whether training is or not. Defaults to True.

        """
        super().__init__()
        self.data_root = Path(data_root)
        self.scenario_dir = self.data_root.joinpath(scenario_dir)
        self.ann_file = Path(ann_file)
        self.agent_types = agent_types
        self.transforms = Compose(transforms)
        self.predict_all_agents = predict_all_agents
        self.training = training

        info_path = self.data_root.joinpath(self.ann_file)
        scenario_info: dict[str, Any] = load_pkl(info_path)
        self.dataset = DatasetName(scenario_info["dataset"])
        self.scenario_ids: list[str] = scenario_info["scenario_ids"]
        msg = f"Total scenarios: {len(self.scenario_ids)}"
        logging.info(msg)

        self._merge_all_iters_to_one_epoch = False
        self.total_epochs = 0

    @property
    def phase(self) -> str:
        """Experiment phase, which returns `"train"` or `"test"`.

        Returns
        -------
            str: Experiment phase.

        """
        return "train" if self.training else "test"

    def merge_all_iters_to_one_epoch(
        self,
        *,
        merge: bool = True,
        epochs: int | None = None,
    ) -> None:
        """Merge all iterations to one epoch.

        Args:
        ----
            merge (bool, optional): Indicates whether to merge. Defaults to True.
            epochs (int | None, optional): Total number of epochs. Defaults to None.

        """
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scenario_id = self.scenario_ids[index]
        info = load_pkl(self.scenario_dir / f"sample_{scenario_id}.pkl")

        ret_info = self.preprocess(info)
        ret_info = self.transforms(ret_info)
        return self.postprocess(ret_info)

    def preprocess(self, info: dict[str, Any]) -> dict[str, Any]:
        """Run process before transformation.

        Args:
        ----
            info (dict[str, Any]): Source info.

        Returns:
        -------
            dict[str, Any]: Result info.

        """
        scenario = AWMLAgentScenario.from_dict(info["scenario"])
        static_map = AWMLStaticMap.from_dict(info["static_map"])

        ret_info: dict[str, Any] = {}
        ret_info["scenario_id"] = np.array([scenario.scenario_id])
        ret_info["scenario"] = scenario
        ret_info["static_map"] = static_map
        # agent types to be considered in this experiment
        ret_info["agent_types"] = self.agent_types
        ret_info["predict_all_agents"] = self.predict_all_agents

        return ret_info

    @abstractmethod
    def postprocess(self, info: dict) -> dict:
        """Run process after transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Result info.

        """
        ...

    @abstractmethod
    def generate_prediction(
        self,
        pred_scores: Tensor,
        pred_trajs: Tensor,
        batch_meta: dict,
    ) -> list[EvaluationData]:
        """Generate prediction output.

        Args:
        ----
            pred_scores (Tensor): Predicted score.
            pred_trajs (Tensor): Predicted trajectory.
            batch_meta (dict): Metadata.

        Returns:
        -------
            list[EvaluationData]: List of EvaluationData instances for each batch scenario.

        """
        ...

    def evaluate(self, eval_data: list[EvaluationData]) -> tuple[str, dict]:
        """Evaluate predictions prepared by `generate_prediction()`.

        Args:
        ----
            eval_data (list[EvaluationData]): List of predictions formatted to be evaluated.

        Returns:
        -------
            tuple[str, dict]: Evaluation result.

        """
        try:
            eval_topk = eval_data[0].num_mode
        except Exception:  # noqa: BLE001
            logging.warning("Failed to parse the number of modes, use 6")
            eval_topk = 6

        from awml_pred.evaluation import common_evaluation, evaluation_result_as_table

        metric_results = common_evaluation(eval_data, self.agent_types)
        metric_result_str = evaluation_result_as_table(metric_results, self.agent_types)
        return metric_result_str, metric_results

    def collate_batch(
        self,
        batch_list: list[dict[str, Any]],
    ) -> tuple[dict[str, Tensor | list[Tensor] | None], dict[str, Any]]:
        """Collate list of data in batch.

        Args:
        ----
        batch_list (list[dict[str, Any]]): List of stacked batch data.

        Returns:
        -------
            tuple[dict[str, Tensor], dict[str, Any]]: Model inputs and meta data.

        """
        assert len(batch_list) > 0, "At least one data should be loaded."
        batch_input = self.collate_batch_input(batch_list)
        batch_meta = self.collate_batch_meta(batch_list)
        return batch_input, batch_meta

    @abstractmethod
    def collate_batch_input(
        self,
        batch_list: list[dict[str, Any]],
    ) -> dict[str, Tensor | list[Tensor] | None]:
        """Collate model inputs from list of stacked batch data.

        Args:
        ----
        batch_list (list[dict[str, Any]]): List of stacked batch data.

        Returns:
        -------
            dict[str, Tensor | list[Tensor] | None]: Model inputs.

        """

    @abstractmethod
    def collate_batch_meta(self, batch_list: list[dict]) -> dict[str, Any]:
        """Collate metadata from list of stacked batch data.

        Args:
        ----
        batch_list (list[dict[str, Any]]): List of stacked batch data.

        Returns:
        -------
            dict[str, Any]: Inputs meta data.

        """
