import logging
from typing import Optional


import c3d
import numpy as np

from mocap_popy.models.marker_trajectory import MarkerTrajectory

LOGGER = logging.getLogger(__name__)


def get_reader(c3d_path: str):
    """!Get a C3D reader."""
    return c3d.Reader(open(c3d_path, "rb"))


def get_point_labels(reader: c3d.Reader) -> list[str]:
    """!Get point labels from a C3D file.

    @param reader C3D reader.
    """
    return [pl.replace(" ", "") for pl in reader.point_labels]


def get_frames(reader: c3d.Reader) -> int:
    """!Get frame numbers from a C3D file.

    @param reader C3D reader.
    """
    frames = []
    for frame, _, _ in reader.read_frames():
        frames.append(frame)
    return frames


def get_marker_positions(
    reader: c3d.Reader,
    marker_labels: Optional[list] = None,
    unit_scale: float = 1,
    start_frame: int = 0,
    end_frame: int = -1,
) -> tuple:
    """!Get marker positions and (estimated) residuals from a C3D file.

    Start and end frames are relative to trial start and end (e.g. start may not be 0 if trial is trimmed)

    @param reader C3D reader.
    @param marker_labels Marker labels (optional). Uses all markers if not provided.
    @param unit_scale Unit scale factor. Default is 1 (mm).
    @param start_frame Start frame (inclusive). Default is 0.
    @param end_frame End frame (exclusive). Default is -1 (all frames).

    @return positions, residuals Each a dict of {marker: list of positions/residuals}.
    @note Positions in each frame are returned in the form (x, y, z).
    """

    point_labels = get_point_labels(reader)
    if marker_labels is not None:
        missing_markers = [ml for ml in marker_labels if ml not in point_labels]
        if missing_markers:
            LOGGER.warning(f"Data is missing the following markers: {missing_markers}")

    marker_labels = marker_labels or point_labels
    marker_indices = [
        point_labels.index(ml) for ml in marker_labels if ml in point_labels
    ]

    positions = {m: [] for m in marker_labels}
    residuals = {m: [] for m in marker_labels}
    for frame, points, _ in reader.read_frames():
        if end_frame >= 0 and frame >= end_frame:
            break

        if frame < start_frame:
            continue

        pos = points[marker_indices, :3] * unit_scale
        res = points[marker_indices, 3]
        for i, label in enumerate(marker_labels):
            positions[label].append(tuple(pos[i]))
            residuals[label].append(res[i])

    return positions, residuals


def get_marker_trajectories(
    reader: c3d.Reader,
    marker_labels: Optional[list] = None,
    unit_scale: float = 1,
    start_frame: int = 0,
    end_frame: int = -1,
) -> tuple:
    """!Get marker trajectories (object) for each marker in a C3D file.

    Assumes markers with positions of (0, 0, 0) are missing.
    Start and end frames are relative to trial start and end (e.g. start may not be 0 if trial is trimmed)

    @param reader C3D reader.
    @param marker_labels Marker labels (optional). Returns trajectories for all markers if not provided.
    @param unit_scale Unit scale factor. Default is 1 (mm).
    @param start_frame Start frame (inclusive). Default is 0.
    @param end_frame End frame (exclusive). Default is -1 (all frames).

    @return dict of {marker: MarkerTrajectory}.
    """

    point_labels = get_point_labels(reader)
    if marker_labels is not None:
        missing_markers = [ml for ml in marker_labels if ml not in point_labels]
        if missing_markers:
            LOGGER.warning(f"Data is missing the following markers: {missing_markers}")

    marker_labels = marker_labels or point_labels
    marker_indices = [
        point_labels.index(ml) for ml in marker_labels if ml in point_labels
    ]

    trajectories = {m: MarkerTrajectory([], [], [], []) for m in marker_labels}
    for frame, points, _ in reader.read_frames():
        if end_frame >= 0 and frame >= end_frame:
            break

        if frame < start_frame:
            continue

        pos = points[marker_indices, :3] * unit_scale
        for i, label in enumerate(marker_labels):
            trajectories[label].x.append(pos[i, 0])
            trajectories[label].y.append(pos[i, 1])
            trajectories[label].z.append(pos[i, 2])
            trajectories[label].exists.append(np.all(pos[i] != 0))

    return trajectories


def check_point_consistency(reader: c3d.Reader):
    """!Just making sure all frames have the same number of points."""
    num_points = 0
    for frame, points, _ in reader.read_frames():
        if num_points == 0:
            num_points = points.shape[0]

        if num_points != points.shape[0]:
            raise ValueError(f"Frame {frame} has {points.shape[0]} points.")
