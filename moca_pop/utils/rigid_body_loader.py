"""!
    Utility functions for loading rigid bodies

    @author C. McCarthy
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from moca_pop.config import regex
from moca_pop.models.rigid_body import RigidBody, Node
from moca_pop.utils import vsk_parser, model_template_loader, plot_utils

LOGGER = logging.getLogger("RigidBodyLoader")


def get_rigid_bodies_from_vsk(
    vsk_fp: str,
    ignore_symmetry: bool = False,
) -> dict[str, RigidBody]:
    """!Returns a dictionary of rigid bodies from a VSK file.

    Rigid bodies must have a corresponding json template.

    @param vsk_fp Path to VSK file.
    """

    model, bodies, parameters, _, _ = vsk_parser.parse_vsk(vsk_fp)

    markers_positions = vsk_parser.parse_marker_positions_by_segment(parameters)

    rigid_bodies: dict[str, RigidBody] = {}
    for body in bodies:
        asym_body_name = body
        side = None
        if not ignore_symmetry:
            try:
                side, asym_body_name, sep = regex.parse_symmetrical_component(body)
            except ValueError:
                LOGGER.warning(f"Body {body} is not a symmetrical component.")

        body_template = model_template_loader.load_template_from_json(
            model, "rigid_body", asym_body_name
        )
        if body_template is None:
            LOGGER.error(f"No json template found for body {body}.")
            continue

        body_template.name = body
        # if not ignore_symmetry and side:
        #     pre =
        #     body_template.add_symmetry_prefix(side)
        if body in markers_positions:
            numeric_marker_pos = {
                m: v for m, v in markers_positions[body].items() if m.isnumeric()
            }
            formated_marker_pos = {}
            if numeric_marker_pos:
                if not ignore_symmetry:
                    asym_n_marker_mapping = {}
                    for nm in numeric_marker_pos:
                        try:
                            side, asym_m, sep = regex.parse_symmetrical_component(nm)
                            asym_n_marker_mapping[asym_m] = f"{side}{sep}{asym_m}"
                        except ValueError:
                            asym_m = nm
                            continue
                    body_template.remap_markers(asym_n_marker_mapping)
                formated_marker_pos = body_template.convert_vsk_marker_positions(
                    numeric_marker_pos
                )

            nonumeric_marker_pos = {
                m: v for m, v in markers_positions[body].items() if not m.isnumeric()
            }
            if not ignore_symmetry:
                asym_nn_marker_mapping = {}
                for nnm in nonumeric_marker_pos:
                    try:
                        side, asym_m, sep = regex.parse_symmetrical_component(nnm)
                        asym_nn_marker_mapping[asym_m] = f"{side}{sep}{asym_m}"
                    except ValueError:
                        continue
                body_template.remap_markers(asym_nn_marker_mapping)

            pos = {
                **nonumeric_marker_pos,
                **formated_marker_pos,
            }

            nodes = [Node(name, p) for name, p in pos.items()]

        else:
            LOGGER.warning(f"Body {body} not found in VSK markers positions.")
            nodes = None

        rigid_body = body_template.to_rigid_body(nodes=nodes)
        rigid_bodies[body] = rigid_body

    return rigid_bodies


def plot_rigid_bodies_on_figure(
    fig: plt.Figure,
    rigid_bodies: list[RigidBody],
    plane: str = "xy",
    subplot_positions: list[int] = None,
    color_palette: list[str] = None,
    ax_margins: tuple[float, float, float] = (1, 0.5, 0.5),
    ax_units: list[str] = None,
) -> list[plt.Axes]:
    """!Plot rigid bodies on a figure.

    @param fig Figure to plot on.
    @param rigid_bodies List of rigid bodies to plot.
    @param plane Plane to plot on.
    @param subplot_positions List of subplot positions. (i.e. [121, 122])
    @param color_palette List of colors to use for each rigid body.
    @param ax_margins Tuple of axis margins, as percentages of the data range.
    @param ax_units List of axis units. Default is "mm" for all axes.
    """
    subplot_positions = subplot_positions or [121, 122]
    color_palette = color_palette or ["b", "r", "g", "c", "m", "y", "k"]

    ax_units = ax_units or ["mm"] * len(subplot_positions)
    axs: list[plt.Axes] = []
    for i in subplot_positions:
        ax = plot_utils.add_axis_using_plane(
            fig, plane, subplot_position=i, axis_unit="mm"
        )
        axs.append(ax)

    for i, rb in enumerate(rigid_bodies):
        ax = axs[i]
        c = color_palette[i]
        rb.draw_nodes_on(
            ax, plane, include_labels=True, color=c, text_kwargs={"fontsize": 10}
        )
        rb.draw_segments_on(ax, plane, color=c)

        coords = np.array([n.position for n in rb.nodes])

        ax.set_title(rb.name)

        plot_utils.set_axis_limits(ax, coords, plane, margin=ax_margins)
        plot_utils.set_axes_equal(ax, plane)

    return axs
