# %%
"""Difference resonance web in phase advance (omega = 2 pi Q, in rad).

Draws every difference resonance m wx - n wy = 2 pi k (m, n >= 1) up to
order m + n <= MAX_ORDER in the window WX_RANGE x WY_RANGE [rad]. Low orders are
drawn dark and thick, high orders light and thin.
"""

# %% Imports
from math import gcd

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% Parameters
try:    # figures are saved next to this script
    FIG_DIR = Path(__file__).resolve().parent
except NameError:    # cells run one by one in an interactive window
    FIG_DIR = Path.cwd()
FIG_PREFIX = '00a'
MAX_ORDER = 8
WX_RANGE = (2.0, 3.0)   # omega_x window [rad]
WY_RANGE = (2.0, 3.0)   # omega_y window [rad]
WORKING_POINT = (2.602, 2.602)   # (omega_x, omega_y) [rad]

# Simulation parameters written in the figure names
FIG_TAG = (f'order{MAX_ORDER}_wx{WX_RANGE[0]}-{WX_RANGE[1]}'
           f'_wy{WY_RANGE[0]}-{WY_RANGE[1]}')


# %% Functions
def line_style(order, max_order):
    """Grey level and width: order 2 is black and thick, max_order is light."""
    t = (order - 2) / max(max_order - 2, 1)   # 0 for order 2, 1 for max order
    grey = str(0.75 * t)                       # '0.0' = black, '0.75' = light grey
    lw = 2.5 - 2.0 * t
    return grey, lw


def resonance_lines(max_order, qx_range, qy_range):
    """Yield (order, qx_pts, qy_pts) for each difference resonance
    m Qx - n Qy = k, with m, n >= 1 and k integer (i.e. m wx - n wy = 2 pi k
    with w = 2 pi Q). The order is m + n.

    Triples with a common divisor repeat a lower order line and are skipped.
    """
    qx0, qx1 = qx_range
    for m in range(1, max_order):
        for n in range(1, max_order - m + 1):
            order = m + n
            # Range of k such that the line crosses the window
            corners = [m * qx - n * qy for qx in qx_range for qy in qy_range]
            for k in range(int(np.floor(min(corners))), int(np.ceil(max(corners))) + 1):
                if gcd(gcd(m, n), abs(k)) != 1:
                    continue
                qx = np.array([qx0, qx1])
                yield order, qx, (m * qx - k) / n


def plot_resonance_web(ax, max_order=MAX_ORDER, wx_range=WX_RANGE, wy_range=WY_RANGE):
    """Lines are computed in tunes (Q = omega / 2 pi) and drawn in omega [rad]."""
    qx_range = tuple(w / (2 * np.pi) for w in wx_range)
    qy_range = tuple(w / (2 * np.pi) for w in wy_range)
    # Highest orders first, so the low orders are drawn on top
    lines = sorted(resonance_lines(max_order, qx_range, qy_range), key=lambda l: -l[0])
    for order, qx, qy in lines:
        grey, lw = line_style(order, max_order)
        ax.plot(2 * np.pi * qx, 2 * np.pi * qy, color=grey, lw=lw, zorder=-order)

    ax.set_xlim(wx_range)
    ax.set_ylim(wy_range)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\omega_x$ [rad]')
    ax.set_ylabel(r'$\omega_y$ [rad]')
    ax.set_title(f'Difference resonances up to order {max_order}')

    handles = [Line2D([], [], color=line_style(o, max_order)[0],
                      lw=line_style(o, max_order)[1], label=f'order {o}')
               for o in range(2, max_order + 1)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)


# %% Plot
fig, ax = plt.subplots(figsize=(7.5, 6))
plot_resonance_web(ax)
ax.axvline(WORKING_POINT[0], color="tab:blue", ls="--", lw=1.2, zorder=1)
ax.plot(*WORKING_POINT, "r*", ms=14, zorder=2, label=f"working point {WORKING_POINT}")
ax.legend(handles=ax.get_legend().legend_handles + [ax.get_lines()[-1]],
          loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / f'{FIG_PREFIX}_resonance_web_{FIG_TAG}.png', dpi=200)
plt.show()

# %%
