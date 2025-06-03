# Autoware MTR Python Package [Archieved, please visit https://github.com/tier4/autoware_mtr_python for latest version of this repo]

## Overview

The `autoware_mtr_python` package is part of the Autoware Universe project. This package provides functionalities for the MTR (Motion Transformer) node implemented in `mtr_node.py`.  For more information about the MTR model, please refer to the original paper publication: [Motion Transformer with Global Intention Localization and Local Movement Refinement]{https://arxiv.org/abs/2209.13508} by Shi et. al.
Furthermore, this project takes many of its tools from the [AWMLprediction repository]{https://github.com/tier4/AWMLprediction}.

## Main Functionality

The `mtr_node.py` script is responsible for detecting and recognizing multiple agents in the vehicle's surroundings, including the ego vehicle itself. It generates a predicted trajectory of the ego vehicle based on its and other agents current and past states.

### Features

- **Trajectory Prediction**: Predicts the future trajectory of the ego vehicle based on its past states and the surrounding environment.
- **Lanelet2 Integration**: Utilizes Lanelet2 map data for spatial context and localization.
- **Agent Interaction Modeling**: Considers the behavior of surrounding agents to refine ego vehicle trajectory predictions.
- **ROS2 Integration**: Subscribes to odometry and tracked object topics and publishes predicted trajectories and objects.
- **Debugging Tools**: Publishes debug markers for visualizing map polylines and predicted trajectories.

---

## Model Inputs

The MTR node subscribes to Autoware's odometry and tracked object topics to gather information about surrounding agents and the environment. It also leverages Lanelet2 map data to localize the ego vehicle and nearby agents.

The table below outlines the input dictionary passed to the MTR model. Note that this node is used solely as a **trajectory generator** for the ego vehicle, so the **batch size is always 1**.

| Key                      | Description                                             | Shape                  | Dimensions Explained                                                  | Data Type         |
|--------------------------|---------------------------------------------------------|-------------------------|------------------------------------------------------------------------|-------------------|
| `obj_trajs`              | Embedded past trajectories of agents                   | `[B, A, T, D]`          | B: batch size, A: agents, T: timesteps, D: features                    | `torch.Tensor`    |
| `obj_trajs_mask`         | Mask indicating valid trajectory steps per agent       | `[B, A, T]`             | B: batch size, A: agents, T: timesteps                                 | `torch.Tensor`    |
| `map_polylines`          | Encoded map polylines (e.g., lanes, crosswalks)        | `[B, L, P, Dl]`         | B: batch size, L: polylines, P: points per polyline, Dl: features      | `torch.Tensor`    |
| `map_polylines_mask`     | Mask for valid polyline points                         | `[B, L, P]`             | B: batch size, L: polylines, P: points per polyline                    | `torch.Tensor`    |
| `map_polylines_center`   | Center coordinates or attributes of each polyline      | `[B, L, C]`             | B: batch size, L: polylines, C: center dimensions (e.g., x, y, z)      | `torch.Tensor`    |
| `obj_trajs_last_pos`     | Final (x, y, yaw) position of each agent               | `[B, A, 3]`             | B: batch size, A: agents, 3: position and orientation                  | `torch.Tensor`    |
| `intention_points`       | Candidate goal or waypoint positions for the ego agent | `[B, N, 2]`             | B: batch size, N: number of candidates, 2: (x, y) coordinates          | `torch.Tensor`    |
| `track_index_to_predict` | Index of the agent to generate trajectory for          | `[B]`                   | B: batch size                                                          | `torch.IntTensor` |

**Default Dimensions:**
`B = 1`, `A = variable`, `T = 11`, `D = 29`, `L = 768`, `P = 20`, `Dl = 9`, `C = 3`, `N = 64`

---

### Details of `obj_trajs`

The `obj_trajs` tensor contains target-centric embedded trajectories for all agents, where each trajectory is expressed in the frame of the target (i.e., the ego vehicle). Each feature vector along the last dimension (`D = 29`) is composed of:

- **Position**: (x, y, z) → 3 dims
- **Size**: width, length, height → 3 dims
- **One-hot encoding**:
  - Agent type (e.g., car, pedestrian, bike) → 3 dims
  - Flags:
    - Is target of prediction → 1 dim
    - Is ego vehicle → 1 dim
- **Timestamps**:
  - one hot encoding for the timestamps order → 11 dims
  - Actual timestamp value → 1 dim
- **Yaw embedding**: sin(yaw), cos(yaw) → 2 dims
- **Velocity**: vx, vy → 2 dims
- **Acceleration**: ax, ay → 2 dims

**Total = 29 features**

---

### Details of `obj_trajs_mask`

A binary mask with the same agent and timestep structure as `obj_trajs`. A value of `1` indicates valid data at a timestep; `0` marks padded or missing data (e.g., for agents that were not yet observed or were missing from perception).

---

### Details of `map_polylines`

This tensor encodes vectorized map features, such as lane centerlines or crosswalk edges. Each polyline is broken into `P = 20` points, and each point has `Dl = 9` features which may include:

- Position (x, y, z) (3)
- 3D normalized direction (dx,dy,dz) (3)
- Lane type or semantic ID (1)
- Position of next point in the polyline (x,y) (2)

Used to provide spatial context to the model. The image below shows a debug visualization of the polylines, color coded for different polyline types.

![map polylines](./images/MTR-polyline.svg)

---

### Details of `map_polylines_mask`

This mask indicates which of the `P` points per polyline are valid. This is useful since not all polylines are exactly 20 points long, and padding with zeroes is used to fill the remaining points.

---

### Details of `map_polylines_center`

Each polyline is assigned a center point, typically its geometric midpoint or representative feature. The center includes `C = 3` values (x, y, z). This helps with spatial aggregation and efficient lookup.

---

### Details of `obj_trajs_last_pos`

This tensor gives the final position and yaw of each agent in the past observation window. Used to initialize future trajectory prediction. The format is `[x, y, z]`.

---

### Details of `intention_points`

These are candidate goal locations for the ego agent. Used by the model to condition the predicted future trajectory. Each point is a 2D coordinate `(x, y)`, and there are `N = 64` candidates.

---

### Details of `track_index_to_predict`

An index tensor used to specify which agents in the input should have their future trajectories predicted. In the ego-only case, this is simply `[0]`.

---

## Outputs

The MTR node predicts 6 modes/trajectories with scores, each prediction consists of 80 points (at approximately 0.1 seconds interval) with position (x,y) and velocity (vx,vy). Further logic is implemented to give a yaw value to each point, which is necessary for Autoware's control to operate.

The picture below shows a representation of the predicted modes for the ego vehicle. It is up to a selector or some other logic implemented in a separate node to select which mode will be used as input for Autoware's control.

![mtr modes](./images/MTR-modes.svg)

## Configuration Options

The behavior of the MTR node can be customized using the configuration file `mtr_ego_node.param.yaml`. Below is a detailed explanation of the available parameters:

| Parameter Name                | Description                                                                 | Default Value                  |
|-------------------------------|-----------------------------------------------------------------------------|--------------------------------|
| `num_timestamp`               | Number of past frames to consider for trajectory prediction.               | `11`                          |
| `timestamp_threshold`         | Maximum allowed timestamp difference (in microseconds) for valid data (currently unused).     | `10000000000000.0`            |
| `score_threshold`             | Minimum score threshold for predicted trajectories to be considered valid. | `0.0`                         |
| `labels`                      | List of agent types to consider (e.g., `VEHICLE`, `PEDESTRIAN`).            | `["VEHICLE"]`                 |
| `ego_dimensions`              | Dimensions of the ego vehicle `[length, width, height]`.                   | `[4.0, 2.0, 1.7]`             |
| `propagate_future_states`     | Whether to propagate future ego states for trajectory prediction.           | `false`                       |
| `add_left_bias_history`       | Whether to add a left-biased trajectory for the ego vehicle.                | `false`                       |
| `add_right_bias_history`      | Whether to add a right-biased trajectory for the ego vehicle.               | `false`                       |
| `publish_debug_polyline_map`  | Whether to publish debug markers for map polylines.                        | `false`                       |
| `future_state_propagation_sec`| Duration (in seconds) for propagating future ego states.                    | `3.0`                         |
| `checkpoint_path`             | Path to the pre-trained model checkpoint file.                             | `$(var data_path)/mtr_best.pth`|
| `model_config`                | Path to the model configuration file.                                      | `$(find-pkg-share autoware_mtr_python)/config/mtr.yaml` |
| `lanelet_file`                | Path to the Lanelet2 map file.                                              | `$(find-pkg-share autoware_mtr_python)/config/odaiba.lanelet2_map.osm` |
| `intention_point_file`        | Path to the intention point file for goal localization.                     | `$(find-pkg-share autoware_mtr_python)/data/cluster64_dict.pkl` |

### Key Parameters

1. **`num_timestamp`**: Determines how many past frames are used to predict the future trajectory. A higher value provides more historical context but increases computational cost.

2. **`ego_dimensions`**: Specifies the physical dimensions of the ego vehicle. This is critical for accurate trajectory prediction and collision avoidance.

3. **`propagate_future_states`**: If enabled, the node will predict future ego states based on the current trajectory and use them for trajectory generation.

4. **`add_left_bias_history` and `add_right_bias_history`**: These options allow the node to generate additional trajectories with a left or right steering bias. This is useful for scenarios where the ego vehicle might need to avoid obstacles or change lanes.

5. **`publish_debug_polyline_map`**: Enables the visualization of map polylines in RViz for debugging purposes. This is helpful for verifying the map data used in predictions.

6. **`checkpoint_path`**: Points to the pre-trained model file used for trajectory prediction. Ensure this file is available and correctly specified.

7. **`lanelet_file`**: Specifies the Lanelet2 map file used for spatial context. This file should match the environment where the vehicle operates.

8. **`intention_point_file`**: Contains pre-defined goal points for the ego vehicle. These points are used to condition the trajectory predictions.

---

## Usage

To run the MTR node, execute the following command:

```bash
ros2 launch autoware_mtr_python mtr_ego_python.launch.xml
```

Ensure that the necessary perception data topics are being published for the node to process.

## Dependencies

- Odometry data (Ego Odometry)
- Perception data input (tracked objects)
