"""!
    MarkerTrajectory class definition
    ================================

    This module contains the definition of the MarkerTrajectory class. 
    It provides an interface between the Vicon Nexus python API and our internal data structures.

    @author C. McCarthy
"""

import dataclasses

import numpy as np

from mocap_popy.models.rigid_body import Node


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
        return Node(
            marker=marker,
            position=np.array([self.x[frame], self.y[frame], self.z[frame]]),
            exists=self.exists[frame],
        )

    def get_frame(self, frame: int) -> MarkerTrajectoryAtFrame:
        if frame < 0 or frame >= len(self.x):
            raise ValueError(f"Frame {frame} out of range.")

        return MarkerTrajectoryAtFrame(
            x=self.x[frame],
            y=self.y[frame],
            z=self.z[frame],
            exists=self.exists[frame],
        )
