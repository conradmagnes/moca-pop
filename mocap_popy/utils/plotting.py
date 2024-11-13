import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorsys


def get_binary_signal_edges(signal, pre_post="post"):
    """!Get the edges of a binary signal.

    @param signal Binary signal.
    @param pre_post Whether to return the pre or post edge of the signal.
    """
    signal_cats = signal.copy()
    if isinstance(signal_cats, list):
        signal_cats = np.array(signal_cats)

    if signal_cats.dtype != "int":
        signal_cats = signal_cats.astype(int)
    signal_cats[signal_cats == 0] = -1

    signal_diff = np.diff(signal_cats)
    lead_edges = np.where(signal_diff == 2)[0]
    lag_edges = np.where(signal_diff == -2)[0]

    if pre_post == "post":
        lead_edges += 1
        lag_edges += 1

    if signal[0] == 1:
        lead_edges = np.insert(lead_edges, 0, 0)

    if signal[-1] == 1 and lead_edges[-1] != len(signal):
        lag_edges = np.append(lag_edges, len(signal) - 1)

    return lead_edges, lag_edges


def draw_shaded_regions(
    ax: plt.Axes, x_starts, x_ends, color="gray", alpha=0.5, ydims=(0, 1), label=None
) -> None:
    """! Draw shaded regions on the plot.

    @param ax Axis to draw on.
    @param x_starts List of start x-coordinates.
    @param x_ends List of end x-coordinates.
    @param color Color of the shaded regions.
    @param alpha Transparency of the shaded regions.
    @param ydims Tuple of shaded y lims, as fractions of the total y-axis limits.
    """
    for i, (start, end) in enumerate(zip(x_starts, x_ends)):
        ax.axvspan(
            start,
            end,
            ymin=ydims[0],
            ymax=ydims[1],
            color=color,
            alpha=alpha,
            label=label if i == 0 else None,
        )


def setup_figure_using_plane(plane: str, figsize=(10, 10), axis_unit: str = ""):
    fig = plt.figure(figsize=figsize)
    if len(plane) == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    ax.set_xlabel(" ".join([plane[0], axis_unit]))
    ax.set_ylabel(" ".join([plane[1], axis_unit]))
    if len(plane) == 3:
        ax.set_zlabel(" ".join([plane[2], axis_unit]))

    return fig, ax


def set_axis_scale_using_plane(ax, plane):
    if len(plane) == 3:
        set_3D_axes_equal(ax)
    else:
        ax.set_aspect("equal")


def set_3D_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Args:
        ax (mpl_toolkits.mplot3d.Axes3D): A matplotlib 3D axis object.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.ptp(limits, axis=1))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def color_to_hue(color_name):
    """
    Converts a named color to its hue value.

    @param color_name A string representing the color (e.g., 'blue', 'tab:blue')
    @return hue (0.0 to 1.0)
    """
    rgb = mpl.colors.to_rgb(color_name)
    hsv = colorsys.rgb_to_hsv(*rgb)
    return hsv[0]


def generate_colormap(color_name, num_colors, vary="saturation"):
    """
    Generates a list of colors with varying saturation or value based on the given color name.

    @param color_name A string representing the base color (e.g., 'blue', 'tab:blue')
    @param num_colors The number of colors to generate
    @param vary 'saturation' or 'value', to specify what to vary (default is 'saturation')
    @return A list of RGB tuples representing the colors
    """
    assert vary in [
        "saturation",
        "value",
    ], "vary must be either 'saturation' or 'value'"

    hue = color_to_hue(color_name)

    base_saturation = 1.0
    base_value = 1.0

    colors = []
    for i in range(num_colors):
        if vary == "saturation":
            saturation = base_saturation * (i + 1) / num_colors
            value = base_value
        elif vary == "value":
            saturation = base_saturation
            value = base_value * (i + 1) / num_colors

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))

    return colors
