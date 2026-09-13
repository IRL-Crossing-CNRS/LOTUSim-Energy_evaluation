import matplotlib.pyplot as plt
import numpy as np


def plot_wake_field(X, Y, U_field, turbines, title="Blended wake velocity field"):
    """Simple top-view contour plot for the blended wake field."""
    fig, ax = plt.subplots(figsize=(9, 5))
    contour = ax.contourf(X, Y, U_field, levels=30)
    fig.colorbar(contour, ax=ax, label="Wind speed (m/s)")

    x_t = [t[2] for t in turbines]  # downstream coordinate on plot x-axis
    y_t = [t[0] for t in turbines]  # lateral coordinate on plot y-axis
    ax.scatter(x_t, y_t, s=80, edgecolors="black")

    for i, (x, _y, z) in enumerate(turbines):
        ax.text(z + 10, x + 10, f"T{i}")

    ax.set_xlabel("Downstream distance z (m)")
    ax.set_ylabel("Lateral distance x (m)")
    ax.set_title(title)
    ax.axis("equal")
    return fig, ax
