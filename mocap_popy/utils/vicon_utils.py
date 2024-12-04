"""!
    Vicon Nexus SDK utility functions
    ==================================

    This module contains utility functions for built on top of the Vicon Nexus SDK.

    @author C. McCarthy
"""

from viconnexusapi import ViconNexus
from mocap_popy.models.marker_trajectory import MarkerTrajectory


def get_trial_frames(vicon: ViconNexus.ViconNexus) -> list[int]:
    """!Get the frames of the active trial in Vicon Nexus.

    @param vicon Vicon Nexus instance
    @return List of frame numbers
    """
    trial_range = vicon.GetTrialRange()
    return list(range(trial_range[0], trial_range[1] + 1))


def get_marker_trajectories(
    vicon: ViconNexus.ViconNexus, subject_name: str
) -> dict[str, MarkerTrajectory]:
    """!Get marker trajectories for a subject from Vicon Nexus.

    @param vicon Vicon Nexus instance
    @param subject_name Name of the subject
    @return Dictionary of marker trajectories
    """
    markers = vicon.GetMarkerNames(subject_name)
    marker_trajectories: dict[str, MarkerTrajectory] = {}
    for m in markers:
        x, y, z, e = vicon.GetTrajectory(subject_name, m)
        marker_trajectories[m] = MarkerTrajectory(x, y, z, e)
    return marker_trajectories
