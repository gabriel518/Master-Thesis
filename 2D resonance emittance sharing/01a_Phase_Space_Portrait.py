# %%
"""Phase space portraits of the resonant normal form Hamiltonian.

    H(phi1, J1) = delta J1 + alpha12 J1 J2 + alpha11 J1^2
                  + G (m J1)^(m/2) (J2 - n J1)^(n/2) cos(phi1)

with G = J2 = 1 and (m, n) = RESONANCE. H has one degree of freedom and does
not depend on time, so it is conserved: the orbits are its level curves,
drawn with a contour plot instead of being integrated.

The orbits are drawn in the plane X = sqrt(2 J1) cos(phi1),
Y = sqrt(2 J1) sin(phi1). The term (J2 - n J1)^(n/2) is real only for
J1 <= J2 / n, i.e. inside the circle of radius sqrt(2 J2 / n), drawn in red.
The separatrix is drawn in black.
"""

# %% Imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# %% Parameters
try:    # figures are saved next to this script
    FIG_DIR = Path(__file__).resolve().parent
except NameError:    # cells run one by one in an interactive window
    FIG_DIR = Path.cwd()
FIG_PREFIX = '01a'

RESONANCE = (1, 2)    # (m, n)
DELTA = 1.0           # distance to the resonance
ALPHA_11 = 0.5
ALPHA_12 = 0.0
G = 1.0
J2 = 1.0

N_GRID = 400          # grid points per axis
N_LEVELS = 40         # number of level curves drawn

FIG_TAG = f'res{RESONANCE[0]}-{RESONANCE[1]}_delta{DELTA}_a11{ALPHA_11}_a12{ALPHA_12}'


# %% Functions
def hamiltonian(phi1, j1, delta, resonance=RESONANCE,
                alpha11=ALPHA_11, alpha12=ALPHA_12, g=G, j2=J2):
    """H(phi1, J1); NaN outside the allowed domain 0 <= J1 <= J2 / n."""
    m, n = resonance
    j1 = np.where((j1 >= 0) & (j1 <= j2 / n), j1, np.nan)
    return (delta * j1 + alpha12 * j1 * j2 + alpha11 * j1**2
            + g * (m * j1) ** (m / 2) * (j2 - n * j1) ** (n / 2) * np.cos(phi1))


def separatrix_level(delta, resonance=RESONANCE,
                     alpha11=ALPHA_11, alpha12=ALPHA_12, j2=J2):
    """Value of H on the separatrix: (alpha11/n^2 + alpha12/n) J2^2 + delta/n J2.

    It is H on the boundary J1 = J2/n, where the resonant term vanishes, so
    the level curve also contains that circle; only the part inside the
    allowed domain is the separatrix.
    """
    m, n = resonance
    return (alpha11 / n**2 + alpha12 / n) * j2**2 + delta / n * j2


def plot_portrait(ax, delta):
    """Level curves of H in the (X, Y) plane, separatrix and allowed circle."""
    m, n = RESONANCE
    r_max = np.sqrt(2 * J2 / n)
    xy = np.linspace(-1.05 * r_max, 1.05 * r_max, N_GRID)
    X, Y = np.meshgrid(xy, xy)
    H = hamiltonian(np.arctan2(Y, X), (X**2 + Y**2) / 2, delta)
    levels = np.linspace(np.nanmin(H), np.nanmax(H), N_LEVELS)
    ax.contour(X, Y, H, levels=levels, cmap='viridis', linewidths=0.8)
    ax.contour(X, Y, H, levels=[separatrix_level(delta)], colors='k', linewidths=1.8)
    ax.plot([], [], 'k', lw=1.8, label='separatrix')
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r_max * np.cos(t), r_max * np.sin(t), 'r', lw=1.5, label='allowed circle')
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'delta = {delta}')
    ax.legend(loc='upper right', fontsize=8)


# %% Plot
fig, ax = plt.subplots(figsize=(6, 5.5))
plot_portrait(ax, DELTA)
fig.suptitle(f'Resonance (m, n) = {RESONANCE}, alpha11 = {ALPHA_11}, alpha12 = {ALPHA_12}')
fig.tight_layout()
fig.savefig(FIG_DIR / f'{FIG_PREFIX}_phase_space_portrait_{FIG_TAG}.png', dpi=200)
plt.show()

# %%
