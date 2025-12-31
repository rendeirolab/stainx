import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_3d_bars(z, x, y, dx=0.1, dy=0.1, color=None, alpha=0.8, edgecolor="black", linewidth=0.1, xlabel="x", ylabel="y", zlabel="z", label_fontsize=12, figsize=(10, 8), dpi=50, elev=30, azim=245, save_path=None, show=True):
    """
    Plot 3D bars for benchmark results.

    Args:
        z: 2D array of values to plot (rows=y, cols=x)
        x: Array of x-axis values (will be treated as categorical, equidistant)
        y: Array of y-axis values (will be treated as categorical, equidistant)
        dx, dy: Bar widths
        color: Bar color (default: viridis)
        alpha: Bar transparency
        edgecolor: Bar edge color
        linewidth: Bar edge width
        xlabel, ylabel, zlabel: Axis labels
        label_fontsize: Font size for labels
        figsize: Figure size
        dpi: Figure DPI
        elev, azim: 3D view angles
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    z = np.array(z)
    x = np.array(x)
    y = np.array(y)
    ny, nx = z.shape

    x_labels = x
    y_labels = y
    x_positions = np.arange(nx)
    y_positions = np.arange(ny)
    x_grid, y_grid = np.meshgrid(x_positions, y_positions)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={"projection": "3d"})

    # Set background to white
    fig.patch.set_facecolor("white")
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    z_min = z.min()
    z_max = z.max()
    z_base = z_min
    z_heights = z.ravel() - z_base

    if color is None:
        color = cm.viridis(0.5)

    x_bar_pos = x_grid.ravel() - dx / 2
    y_bar_pos = y_grid.ravel() - dy / 2

    ax.bar3d(x_bar_pos, y_bar_pos, z_base, dx, dy, z_heights, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)

    fixed_margin = 0.5
    ax.set_xlim(x_positions.min() - fixed_margin, x_positions.max() + fixed_margin)
    ax.set_ylim(y_positions.min() - fixed_margin, y_positions.max() + fixed_margin)
    ax.set_zlim(z_min, z_max)

    z_ticks = np.round(np.linspace(z_min, z_max, 6), 1)
    ax.set_zticks(z_ticks)

    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")
    ax.set_zlabel(zlabel, fontsize=label_fontsize, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)

    # Show z-ticks on both sides by manually adding them to the opposite side
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    tick_length = 0.02 * (xlim[1] - xlim[0])

    for tick_val in z_ticks:
        # Add tick marks on the opposite side (at x=max, y=min)
        ax.plot([xlim[1] - tick_length, xlim[1]], [ylim[0], ylim[0]], [tick_val, tick_val], "k-", linewidth=0.8, alpha=0.7)
        # Add tick labels on the opposite side
        ax.text(xlim[1] + tick_length, ylim[0], tick_val, f"{tick_val:.1f}", ha="left", va="center", fontsize=9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_2d_bars(x, y, width=0.25, color=None, alpha=0.8, edgecolor="black", linewidth=0.5, xlabel="x", ylabel="y", label_fontsize=12, figsize=(10, 8), dpi=50, save_path=None, show=True):
    """
    Plot 2D bars for benchmark results.

    Args:
        x: Array of x-axis labels (will be treated as categorical, equidistant)
        y: Array of y-axis values (bar heights)
        width: Bar width (relative to spacing)
        color: Bar color (default: viridis)
        alpha: Bar transparency
        edgecolor: Bar edge color
        linewidth: Bar edge width
        xlabel, ylabel: Axis labels
        label_fontsize: Font size for labels
        figsize: Figure size
        dpi: Figure DPI
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    if len(y) != n:
        raise ValueError(f"x and y must have the same length. Got x: {n}, y: {len(y)}")

    x_labels = x
    x_positions = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set background to white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Generate color if not provided
    if color is None:
        color = cm.viridis(0.5)

    # Plot bars
    ax.bar(x_positions, y, width, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)

    # Set axis limits with margin
    fixed_margin = 0.5
    ax.set_xlim(x_positions.min() - fixed_margin, x_positions.max() + fixed_margin)
    y_min = y.min()
    y_max = y.max()
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - 0.1 * y_range) if y_min >= 0 else y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Set ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    # Format y-ticks
    y_ticks = np.round(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6), 1)
    ax.set_yticks(y_ticks)

    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
