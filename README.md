# Autoware MTR Python Package

## Overview

The `autoware_mtr_python` package is part of the Autoware Universe project, specifically within the perception module. This package provides functionalities for the MTR (Motion Transformer) node implemented in `mtr_node.py`.  For more information about the MTR model, please refer to the original paper publication: [Motion Transformer with Global Intention Localization and Local Movement Refinement]{https://arxiv.org/abs/2209.13508} by Shi et. al.
Furthermore, this project takes many of its tools from the [AWMLprediction repository]{https://github.com/tier4/AWMLprediction}.

## Main Functionality

The `mtr_node.py` script is responsible for detecting and recognizing multiple agents in the vehicle's surroundings, including the ego vehicle itself. It generates a predicted trajectory of the ego vehicle based on its and other agents current and past states.


## Usage

To run the MTR node, execute the following command:

```bash
ros2 launch autoware_mtr_python mtr_ego_python.launch.xml
```

Ensure that the necessary perception data topics are being published for the node to process.

## Dependencies

- Odometry data (Ego Odometry)
- Perception data input (tracked objects)
