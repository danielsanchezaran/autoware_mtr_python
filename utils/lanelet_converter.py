from __future__ import annotations

import logging
import sys

import numpy as np

try:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector
    from lanelet2.routing import RoutingGraph
    from lanelet2.traffic_rules import Locations, Participants
    from lanelet2.traffic_rules import create as create_traffic_rules
except ImportError as e:
    print(e)  # noqa: T201
    sys.exit(1)

from awml_pred.common import uuid
from awml_pred.dataclass import AWMLStaticMap, BoundarySegment, CrosswalkSegment, LaneSegment, Polyline
from awml_pred.datatype import BoundaryType, T4Lane, T4Polyline, T4RoadEdge, T4RoadLine

# cspell: ignore MGRS


def _load_osm(filename: str) -> lanelet2.core.LaneletMap:
    """Load lanelet map from osm file.

    Args:
    ----
        filename (str): Path to osm file.

    Returns:
    -------
        lanelet2.core.LaneletMap: Loaded lanelet map.

    """
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    return lanelet2.io.load(filename, projection)


def _get_lanelet_subtype(lanelet: lanelet2.core.Lanelet) -> str:
    """Return subtype name from lanelet.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        str: Subtype name. Return "" if it has no attribute named subtype.

    """
    if "subtype" in lanelet.attributes:
        return lanelet.attributes["subtype"]
    else:
        return ""


def _get_linestring_type(linestring: lanelet2.core.LineString3d) -> str:
    """Return type name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Type name. Return "" if it has no attribute named type.

    """
    if "type" in linestring.attributes:
        return linestring.attributes["type"]
    else:
        return ""


def _get_linestring_subtype(linestring: lanelet2.core.LineString3d) -> str:
    """Return subtype name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Subtype name. Return "" if it has no attribute named subtype.

    """
    if "subtype" in linestring.attributes:
        return linestring.attributes["subtype"]
    else:
        return ""


def _is_virtual_linestring(line_type: str, line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is virtual.

    Args:
    ----
        line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is `virtual` and subtype is `""`.

    """
    return line_type == "virtual" and line_subtype == ""


def _is_roadedge_linestring(line_type: str, _line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is supported RoadEdge.

    Args:
    ----
        line_type (str): Line type name.
        _line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is contained in T4RoadEdge.

    Note:
    ----
        Currently `_line_subtype` is not used, but it might be used in the future.

    """
    return line_type.upper() in T4RoadEdge.__members__


def _is_roadline_linestring(_line_type: str, line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is supported RoadLine.

    Args:
    ----
        _line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line subtype is contained in T4RoadLine.

    Note:
    ----
        Currently `_line_type` is not used, but it might be used in the future.

    """
    return line_subtype.upper() in T4RoadLine.__members__


def _get_boundary_type(linestring: lanelet2.core.LineString3d) -> BoundaryType:
    """Return the `BoundaryType` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundaryType: BoundaryType instance.

    """
    line_type = _get_linestring_type(linestring)
    line_subtype = _get_linestring_subtype(linestring)
    if _is_virtual_linestring(line_type, line_subtype):
        return T4RoadLine.VIRTUAL
    elif _is_roadedge_linestring(line_type, line_subtype):
        return T4RoadEdge.from_str(line_type)
    elif _is_roadline_linestring(line_type, line_subtype):
        return T4RoadLine.from_str(line_subtype)
    else:
        logging.warning(
            f"[Boundary]: id={linestring.id}, type={line_type}, subtype={line_subtype}, T4RoadLine.VIRTUAL is used.",
        )
        return T4RoadLine.VIRTUAL


def _get_boundary_segment(linestring: lanelet2.core.LineString3d) -> BoundarySegment:
    """Return the `BoundarySegment` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundarySegment: BoundarySegment instance.

    """
    boundary_type = _get_boundary_type(linestring)
    waypoints = np.array([(line.x, line.y, line.z) for line in linestring])
    global_type = T4Polyline.from_str(boundary_type.as_str())
    polyline = Polyline(polyline_type=global_type, waypoints=waypoints)
    return BoundarySegment(linestring.id, boundary_type, polyline)


def _get_speed_limit_mph(lanelet: lanelet2.core.Lanelet) -> float | None:
    """Return the lane speed limit in miles per hour (mph).

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        float | None: If the lane has the speed limit return float, otherwise None.

    """
    kph2mph = 0.621371
    if "speed_limit" in lanelet.attributes:
        # NOTE: attributes of ["speed_limit"] is str
        return float(lanelet.attributes["speed_limit"]) * kph2mph
    else:
        return None


def _get_left_and_right_linestring(
    lanelet: lanelet2.core.Lanelet,
) -> tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]:
    """Return the left and right boundaries from lanelet.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]: Left and right boundaries.

    """
    return lanelet.leftBound, lanelet.rightBound


def _is_intersection(lanelet: lanelet2.core.Lanelet) -> bool:
    """Check whether specified lanelet is intersection.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        bool: Return `True` if the lanelet has an attribute named `turn_direction`.

    """
    return "turn_direction" in lanelet.attributes


def _get_left_and_right_neighbor_ids(
    lanelet: lanelet2.core.Lanelet,
    routing_graph: RoutingGraph,
) -> tuple[list[int], list[int]]:
    """Return whether the lanelet has left and right neighbors.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.
        routing_graph (RoutingGraph): RoutingGraph instance.

    Returns:
    -------
        tuple[list[int], list[int]]: Whether the lanelet has (left, right) neighbors.

    """
    left_lanelet = routing_graph.left(lanelet)
    right_lanelet = routing_graph.right(lanelet)
    left_neighbor_id = [left_lanelet.id] if left_lanelet is not None else []
    right_neighbor_id = [right_lanelet.id] if right_lanelet is not None else []
    return left_neighbor_id, right_neighbor_id


def convert_lanelet(filename: str) -> AWMLStaticMap:
    """Convert lanelet (.osm) to map info.

    Note:
    ----
        Currently, following subtypes are skipped:
            walkway

    Args:
    ----
        filename (str): Path to osm file.

    Returns:
    -------
        AWMLStaticMap: Static map data.

    """
    lanelet_map = _load_osm(filename)

    traffic_rules = create_traffic_rules(Locations.Germany, Participants.Vehicle)
    routing_graph = RoutingGraph(lanelet_map, traffic_rules)

    lane_segments: dict[int, LaneSegment] = {}
    crosswalk_segments: dict[int, CrosswalkSegment] = {}
    taken_boundary_ids: list[int] = []
    for lanelet in lanelet_map.laneletLayer:
        lanelet_subtype = _get_lanelet_subtype(lanelet)
        if lanelet_subtype == "":
            continue

        # NOTE: skip walkway because it contains stop_line as boundary
        if T4Lane.contains(lanelet_subtype) and lanelet_subtype != "walkway":
            # lane
            lane_type = T4Lane.from_str(lanelet_subtype)
            lane_waypoints = np.array([(line.x, line.y, line.z) for line in lanelet.centerline])
            global_lane_type = T4Polyline.from_str(lanelet_subtype)
            lane_polyline = Polyline(polyline_type=global_lane_type, waypoints=lane_waypoints)
            is_intersection = _is_intersection(lanelet)
            left_neighbor_ids, right_neighbor_ids = _get_left_and_right_neighbor_ids(lanelet, routing_graph)
            speed_limit_mph = _get_speed_limit_mph(lanelet)

            # road line or road edge
            left_linestring, right_linestring = _get_left_and_right_linestring(lanelet)
            left_boundary = _get_boundary_segment(left_linestring)
            right_boundary = _get_boundary_segment(right_linestring)
            taken_boundary_ids.extend((left_linestring.id, right_linestring.id))

            lane_segments[lanelet.id] = LaneSegment(
                id=lanelet.id,
                lane_type=lane_type,
                polyline=lane_polyline,
                is_intersection=is_intersection,
                left_boundaries=[left_boundary],
                right_boundaries=[right_boundary],
                left_neighbor_ids=left_neighbor_ids,
                right_neighbor_ids=right_neighbor_ids,
                speed_limit_mph=speed_limit_mph,
            )
        elif lanelet_subtype == "crosswalk":
            waypoints = np.array([(poly.x, poly.y, poly.z) for poly in lanelet.polygon3d()])
            polygon = Polyline(polyline_type=T4Polyline.CROSSWALK, waypoints=waypoints)
            crosswalk_segments[lanelet.id] = CrosswalkSegment(lanelet.id, polygon)
        else:
            logging.warning(f"[Lanelet]: {lanelet_subtype} is unsupported and skipped.")
            continue

    boundary_segments: dict[int, BoundarySegment] = {}
    for linestring in lanelet_map.lineStringLayer:
        type_name: str = _get_linestring_type(linestring)
        if (
            T4RoadEdge.contains(type_name) or T4RoadLine.contains(type_name)
        ) and linestring.id not in taken_boundary_ids:
            boundary_segments[linestring.id] = _get_boundary_segment(linestring)

    # generate uuid from map filepath
    map_id = uuid(filename, digit=16)
    return AWMLStaticMap(
        map_id,
        lane_segments=lane_segments,
        crosswalk_segments=crosswalk_segments,
        boundary_segments=boundary_segments,
    )
