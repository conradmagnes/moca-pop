"""!
    Interactive score analyzer helper functions.
    ============================================

    @author C. McCarthy
"""

import numpy as np

from mocap_popy.config import regex
from mocap_popy.models.rigid_body import RigidBody
from mocap_popy.scripts.unassign_rb_markers.scoring import scoringParameters


def refactor_removal_threshold(
    scoring_params: scoringParameters.ScoringParameters, marker_names: list[str]
):
    """!Refactor the removal threshold to be a dictionary with marker names as keys.

    @param scoring_params Scoring parameters object.
    @param marker_names List of marker names.
    """
    if isinstance(scoring_params.removal_threshold, dict):
        removal_threshold_all = 0
        for m in marker_names:
            if m not in scoring_params.removal_threshold:
                scoring_params.removal_threshold[m] = 0
    else:
        removal_threshold_all = scoring_params.removal_threshold
        scoring_params.removal_threshold = {
            m: scoring_params.removal_threshold for m in marker_names
        }

    return scoring_params, removal_threshold_all


def remove_symmetry_from_scoring(scoring_params: scoringParameters.ScoringParameters):
    """!Remove symmetry from the scoring parameters.

    @param scoring_params Scoring parameters object.
    """
    removal_threshold = {}
    for k, v in scoring_params.removal_threshold.items():
        try:
            _, symm_comp = regex.parse_symmetrical_component(k)
        except ValueError:
            symm_comp = k
        removal_threshold[symm_comp] = v

    scoring_params.removal_threshold = removal_threshold
    return scoring_params


def best_fit_transform(target_body: RigidBody, active_body: RigidBody):
    """
    Computes the optimal rigid transformation (rotation + translation) that aligns
    points A with points B using the Kabsch algorithm.

    @param target_body Rigid body to be transformed.
    @param active_body Rigid body to be used as the reference.
    """
    active_nodes = [n for n in active_body.nodes if n.exists]
    active_markers = [n.marker for n in active_nodes]
    active_positions = np.array([n.position for n in active_nodes])
    calibrated_positions = np.array(
        [target_body.get_node(m).position for m in active_markers]
    )

    A = calibrated_positions
    B = active_positions

    # 1. Calculate centroids of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Translate points to have their centroids at the origin
    AA = A - centroid_A
    BB = B - centroid_B

    # 3. Compute the covariance matrix H
    H = AA.T @ BB

    # 4. Compute the Singular Value Decomposition (SVD) of H
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute the optimal rotation matrix R
    R = Vt.T @ U.T

    # Special reflection case to correct improper rotation matrices
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute the optimal translation vector t
    t = centroid_B - R @ centroid_A

    all_calibrated_positions = np.array([n.position for n in target_body.nodes])
    transformed_points = apply_transform(all_calibrated_positions, R, t)
    target_body.update_node_positions(
        {m: p for m, p in zip(target_body.get_markers(), transformed_points)}
    )

    return


def apply_transform(points, R, t):
    """
    Applies the given rigid transformation (rotation + translation) to the points.

    @param points (N, 3) ndarray of points to transform.
    @param R (3, 3) ndarray representing the rotation matrix.
    @param t (3, ) ndarray representing the translation vector.

    @return transformed_points (N, 3) ndarray of transformed points.
    """
    return (R @ points.T).T + t
