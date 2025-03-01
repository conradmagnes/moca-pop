"""!
    MarkerTrajectory class definition
    ================================

    This module contains the definition of the MarkerTrajectory class. 
    It provides an interface between the Vicon Nexus python API and our internal data structures.

    @author C. McCarthy
"""

import dataclasses

import numpy as np

from moca_pop.models.rigid_body import Node
from moca_pop.utils import plot_utils


@dataclasses.dataclass
class MarkerTrajectoryAtFrame:
    x: float
    y: float
    z: float
    exists: bool


@dataclasses.dataclass
class MarkerTrajectory:
    x: list[float]
    y: list[float]
    z: list[float]
    exists: list[bool]

    def generate_node(self, marker, frame: int) -> Node:
        try:
            return Node(
                marker=marker,
                position=np.array([self.x[frame], self.y[frame], self.z[frame]]),
                exists=self.exists[frame],
            )
        except IndexError:
            return Node(marker=marker, position=np.zeros(3), exists=False)

    def get_frame(self, frame: int) -> MarkerTrajectoryAtFrame:
        if frame < 0 or frame >= len(self.x):
            raise ValueError(f"Frame {frame} out of range.")

        return MarkerTrajectoryAtFrame(
            x=self.x[frame],
            y=self.y[frame],
            z=self.z[frame],
            exists=self.exists[frame],
        )

    def get_perc_labeled(self) -> float:
        if len(self.exists) == 0:
            return 0
        return 100 * sum(self.exists) / len(self.exists)

    def get_num_gaps(self) -> int:
        if len(self.exists) == 0:
            return 0

        lead, _ = plot_utils.get_binary_signal_edges(self.exists)
        return len(lead) - 1
