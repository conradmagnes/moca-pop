import logging

from viconnexusapi import ViconNexus

from mocap_popy.models.marker_trajectory import MarkerTrajectory

LOGGER = logging.getLogger("QualityReport")


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


def get_num_gaps(
    marker_trajectories: dict[str, MarkerTrajectory]
) -> tuple[int, dict[str, int]]:
    """!Get the number of gaps in each marker trajectory."""
    return {m: marker_trajectories[m].get_num_gaps() for m in marker_trajectories}


def get_perc_labeled(
    marker_trajectories: dict[str, MarkerTrajectory]
) -> tuple[float, dict[str, float]]:
    """!Get the percentage of labeled frames in each marker trajectory."""
    return {
        m: 100 * marker_trajectories[m].get_perc_labeled() for m in marker_trajectories
    }


def log_gaps(num_gaps: dict[str, int]) -> None:
    """!Log the number of gaps in each marker and the total number of gaps."""
    total_gaps = sum(num_gaps.values())
    sorted_gaps = sorted(num_gaps.items(), key=lambda x: x[1], reverse=True)
    LOGGER.info(f"Number of Gaps\n==========\n")
    LOGGER.info(f"- Total: {total_gaps} -")
    for m, n in sorted_gaps:
        if n > 0:
            LOGGER.info(f"{m}: {n}")


def log_labeled(perc_labeled) -> None:
    """!Log the percentage of labeled frames in each marker and the overall percentage of labeled frames."""
    total_labeled = sum(perc_labeled.values()) / len(perc_labeled)
    sorted_labeled = sorted(perc_labeled.items(), key=lambda x: x[1], reverse=True)
    LOGGER.info(f"\Percent Labeled\n==========\n")
    LOGGER.info(f"- Total: {total_labeled:.2f}% -")
    for m, p in sorted_labeled:
        LOGGER.info(f"{m}: {p:.2f}%")
