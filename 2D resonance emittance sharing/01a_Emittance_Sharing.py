# %%
"""Emittance sharing near a 2D difference resonance.

Many particles are tracked with the 4D Henon-like map

    (x, px, y, py)' = R(omega_x, omega_y) . (x, px + sqrt(beta_x) Re[c z^r],
                                             y, py - sqrt(beta_y) Im[c z^r])

with c = (K_r + i J_r) / r! and z = sqrt(beta_x) x + i sqrt(beta_y) y.
R is a rotation by omega_x in (x, px) and by omega_y in (y, py).

N_PART particles start from the exponential distribution
rho(Jx, Jy) = exp(-Jx/eps_x - Jy/eps_y) / (eps_x eps_y), with uniform phases.
They are tracked for N_TURNS turns while omega_y is varied linearly from
m omega_x / n + delta_max to m omega_x / n - delta_max, so that the
difference resonance m omega_x - n omega_y = 2 pi k (RESONANCE, RESONANCE_K)
is crossed slowly, over the whole simulation.

At the end, the histograms of Jx and Jy are compared with the initial ones:
the mean actions <J> are computed and an exponential is fitted on each
histogram, to measure how the emittance has been shared between the planes.
The figure is saved as emittance_sharing.png.
"""

# %% Imports
from math import factorial

import numpy as np

# %% Parameters
OMEGA_X = 2.602                # horizontal phase advance per turn [rad]
RESONANCE = (1, 2)             # (m, n) of m omega_x - n omega_y = 2 pi k
RESONANCE_K = 0                # integer k of the resonance condition (set the resonance condition in the equivalence class)
DELTA_MAX = 0.1                # largest distance to the resonance [rad]

ORDER_R = 2                    # multipole order r of the kick
K_R = 1.0                      # normal multipole strength K_r
J_R = 0.0                      # skew multipole strength J_r
BETA_X = 1.0
BETA_Y = 1.0

N_TURNS = int(1e6)             # omega_y is swept over the whole simulation
N_PART = int(1e4)
EPSILON_X = 1e-4               # initial emittances (mean actions)
EPSILON_Y = 1e-4
SEED = 123


# %% Functions
def omega_y_from_resonance(omega_x, resonance, delta, k=RESONANCE_K):
    """omega_y at a distance delta from m omega_x - n omega_y = 2 pi k.

    delta = 0 is exactly on the resonance; delta may be an array.
    """
    m, n = resonance
    return (m * omega_x - 2 * np.pi * k) / n + delta


def henon_step(x, px, y, py, omega_x, omega_y,
               order=ORDER_R, k_r=K_R, j_r=J_R, beta_x=BETA_X, beta_y=BETA_Y):
    """One turn of the map for every particle at once (numpy arrays).

    The multipole kick is applied first, then the rotation in each plane.
    omega_x and omega_y may be scalars or arrays broadcastable to x.
    """
    sbx, sby = np.sqrt(beta_x), np.sqrt(beta_y)
    kick = (k_r + 1j * j_r) / factorial(order) * (sbx * x + 1j * sby * y) ** order
    px = px + sbx * kick.real
    py = py - sby * kick.imag

    cx, sx = np.cos(omega_x), np.sin(omega_x)
    cy, sy = np.cos(omega_y), np.sin(omega_y)
    x, px = cx * x + sx * px, -sx * x + cx * px
    y, py = cy * y + sy * py, -sy * y + cy * py
    return x, px, y, py


def initial_distribution(n_part, eps_x, eps_y, rng):
    """rho(Jx, Jy) = exp(-Jx/eps_x - Jy/eps_y) / (eps_x eps_y), uniform phases.

    The density factorises: Jx and Jy are exponential with means eps_x, eps_y.
    """
    jx = rng.exponential(eps_x, n_part)
    jy = rng.exponential(eps_y, n_part)
    phx = rng.uniform(0, 2 * np.pi, n_part)
    phy = rng.uniform(0, 2 * np.pi, n_part)
    x, px = np.sqrt(2 * jx) * np.cos(phx), -np.sqrt(2 * jx) * np.sin(phx)
    y, py = np.sqrt(2 * jy) * np.cos(phy), -np.sqrt(2 * jy) * np.sin(phy)
    return x, px, y, py


def actions(x, px, y, py):
    return (x**2 + px**2) / 2, (y**2 + py**2) / 2


def fit_exponential(j, bins=60):
    """Fit rho(J) = exp(-J/eps)/eps on the histogram by a linear fit of
    log(density) vs J (empty bins skipped, weighted by sqrt(counts)).
    Returns eps_fit, the bin centres and the densities."""
    counts, edges = np.histogram(j, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    density = counts / (counts.sum() * np.diff(edges))
    ok = counts > 0
    slope, _ = np.polyfit(centres[ok], np.log(density[ok]), 1, w=np.sqrt(counts[ok]))
    return -1 / slope


# %% Initial distribution
rng = np.random.default_rng(SEED)
x, px, y, py = initial_distribution(N_PART, EPSILON_X, EPSILON_Y, rng)
jx0, jy0 = actions(x, px, y, py)

# %% Tracking: omega_y goes linearly from m wx/n + delta_max to m wx/n - delta_max
deltas = np.linspace(DELTA_MAX, -DELTA_MAX, N_TURNS)
omegas_y = omega_y_from_resonance(OMEGA_X, RESONANCE, deltas)
for turn in range(N_TURNS):
    x, px, y, py = henon_step(x, px, y, py, OMEGA_X, omegas_y[turn])
    if turn % (N_TURNS // 10) == 0:
        print(f'turn {turn} / {N_TURNS}')

jx, jy = actions(x, px, y, py)
alive = np.isfinite(jx) & np.isfinite(jy)
print(f'{alive.sum()} / {N_PART} particles survived')

# %% Results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, name, j_init, j_fin in [(axes[0], 'x', jx0, jx[alive]),
                                (axes[1], 'y', jy0, jy[alive])]:
    eps_i, eps_f = fit_exponential(j_init), fit_exponential(j_fin)
    bins = np.linspace(0, np.percentile(np.r_[j_init, j_fin], 99.5), 60)
    ax.hist(j_init, bins=bins, density=True, alpha=0.4,
            label=f'initial  <J>={j_init.mean():.2e}, fit eps={eps_i:.2e}')
    ax.hist(j_fin, bins=bins, density=True, alpha=0.4,
            label=f'final  <J>={j_fin.mean():.2e}, fit eps={eps_f:.2e}')
    jj = np.linspace(0, bins[-1], 200)
    ax.plot(jj, np.exp(-jj / eps_i) / eps_i, 'C0-')
    ax.plot(jj, np.exp(-jj / eps_f) / eps_f, 'C1-')
    ax.set_yscale('log')
    ax.set_xlabel(f'$J_{name}$')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
fig.suptitle(f'Crossing of the {RESONANCE} resonance in {N_TURNS} turns, '
             f'$\\omega_x$={OMEGA_X}, $\\delta_{{max}}$={DELTA_MAX:.3f}')
fig.tight_layout()
fig.savefig('emittance_sharing.png', dpi=200)
plt.show()

# %%
