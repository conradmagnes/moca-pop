"""!
    Quality Check Utilities
    =======================

    This module contains utility functions for checking the quality of marker trajectories.

    @author C. McCarthy
"""

import logging


from mocap_popy.models.marker_trajectory import MarkerTrajectory

LOGGER = logging.getLogger("QualityReport")


def get_num_gaps(
    marker_trajectories: dict[str, MarkerTrajectory]
) -> tuple[int, dict[str, int]]:
    """!Get the number of gaps in each marker trajectory."""
    return {m: marker_trajectories[m].get_num_gaps() for m in marker_trajectories}


def get_perc_labeled(
    marker_trajectories: dict[str, MarkerTrajectory]
) -> tuple[float, dict[str, float]]:
    """!Get the percentage of labeled frames in each marker trajectory."""
    return {m: marker_trajectories[m].get_perc_labeled() for m in marker_trajectories}


def log_gaps(num_gaps: dict[str, int]) -> None:
    """Log the number of gaps in each marker and the total number of gaps."""
    total_gaps = sum(num_gaps.values())
    sorted_gaps = sorted(num_gaps.items(), key=lambda x: x[1], reverse=True)

    log_output = []
    log_output.append("\nNumber of Gaps")
    log_output.append("=" * 20)
    log_output.append(f"Total Gaps: {total_gaps}")
    log_output.append("-" * 20)
    log_output.append(f"{'Marker':<20}{'Gaps':>6}")
    log_output.append("-" * 20)
    for marker, gaps in sorted_gaps:
        if gaps > 0:
            log_output.append(f"{marker:<20}{gaps:>6}")
    log_output.append("-" * 20)

    LOGGER.info("\n".join(log_output))


def log_labeled(perc_labeled: dict[str, float]) -> None:
    """Log the percentage of labeled frames in each marker and the overall percentage of labeled frames."""
    total_labeled = (
        sum(perc_labeled.values()) / len(perc_labeled) if len(perc_labeled) > 0 else 0
    )
    sorted_labeled = sorted(perc_labeled.items(), key=lambda x: x[1])

    log_output = []
    log_output.append("\nPercent Labeled")
    log_output.append("=" * 20)
    log_output.append(f"Overall Percentage Labeled: {total_labeled:.2f}%")
    log_output.append("-" * 20)
    log_output.append(f"{'Marker':<20}{'Labeled %':>10}")
    log_output.append("-" * 20)
    for marker, percent in sorted_labeled:
        if percent < 100:
            log_output.append(f"{marker:<20}{percent:>10.2f}%")
    log_output.append("-" * 20)

    LOGGER.info("\n".join(log_output))
