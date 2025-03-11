from __future__ import annotations

from awml_pred.common import uuid
from awml_pred.datatype import AgentType, MapType

AGENT_TYPE_MAPPING: dict[str, AgentType] = {
    # Vehicle
    "vehicle.car": AgentType.VEHICLE,
    "vehicle.emergency (ambulance & police)": AgentType.VEHICLE,
    "vehicle": AgentType.VEHICLE,
    "car": AgentType.VEHICLE,
    "police_car": AgentType.VEHICLE,
    "ambulance": AgentType.VEHICLE,
    # Pedestrian
    "pedestrian.adult": AgentType.PEDESTRIAN,
    "pedestrian.child": AgentType.PEDESTRIAN,
    "pedestrian.police_officer": AgentType.PEDESTRIAN,
    "pedestrian.stroller": AgentType.PEDESTRIAN,
    "pedestrian.personal_mobility": AgentType.PEDESTRIAN,
    "pedestrian.construction_worker": AgentType.PEDESTRIAN,
    "pedestrian.wheelchair": AgentType.PEDESTRIAN,
    "pedestrian": AgentType.PEDESTRIAN,
    "police_officer": AgentType.PEDESTRIAN,
    "stroller": AgentType.PEDESTRIAN,
    "personal_mobility": AgentType.PEDESTRIAN,
    "construction_worker": AgentType.PEDESTRIAN,
    "wheelchair": AgentType.PEDESTRIAN,
    # Cyclist
    "vehicle.bicycle": AgentType.CYCLIST,
    "bicycle": AgentType.CYCLIST,
    # Motorcyclist
    "vehicle.motorbike": AgentType.MOTORCYCLIST,
    "vehicle.motorcycle": AgentType.MOTORCYCLIST,
    "motorbike": AgentType.MOTORCYCLIST,
    "motorcycle": AgentType.MOTORCYCLIST,
    # Large vehicle
    "vehicle.bus": AgentType.LARGE_VEHICLE,
    "vehicle.bus (bendy & rigid)": AgentType.LARGE_VEHICLE,
    "vehicle.trailer": AgentType.LARGE_VEHICLE,
    "vehicle.truck": AgentType.LARGE_VEHICLE,
    "vehicle.construction": AgentType.VEHICLE,
    "trailer": AgentType.LARGE_VEHICLE,
    "bus": AgentType.LARGE_VEHICLE,
    "truck": AgentType.LARGE_VEHICLE,
    "fire_truck": AgentType.LARGE_VEHICLE,
    "semi_trailer": AgentType.LARGE_VEHICLE,
    "tractor_unit": AgentType.LARGE_VEHICLE,
    "construction": AgentType.LARGE_VEHICLE,
    # Static
    "movable_object.trafficcone": AgentType.STATIC,
    "movable_object.traffic_cone": AgentType.STATIC,
    "movable_object.pushable_pullable": AgentType.STATIC,
    "static_object.bicycle_rack": AgentType.STATIC,
}

MAP_TYPE_MAPPING: dict[str, MapType] = {
    "road": MapType.ROADWAY,
    "highway": MapType.ROADWAY,
    "road_shoulder": MapType.ROADWAY,
    "bicycle_lane": MapType.BIKE_LANE,
    "dashed": MapType.DASHED,
    "solid": MapType.SOLID,
    "dashed_dashed": MapType.DOUBLE_DASH,
    "virtual": MapType.UNKNOWN,
    "road_border": MapType.SOLID,
    "crosswalk": MapType.CROSSWALK,
    "unknown": MapType.UNKNOWN,
}

T4_LANE: tuple[str, ...] = ("road", "highway", "road_shoulder", "bicycle_lane")
T4_ROADLINE: tuple[str, ...] = ("dashed", "solid", "dashed_dashed", "virtual")
T4_ROADEDGE: tuple[str, ...] = ("road_border",)

EGO_ID = uuid("AV")
