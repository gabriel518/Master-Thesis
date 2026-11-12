# %%
"""phi1 and dphi1/dt along an orbit of the resonant normal form.

    H(phi1, J1) = delta J1 + alpha12 J1 J2 + alpha11 J1^2
                  + G (m J1)^(m/2) (J2 - n J1)^(n/2) cos(phi1)

Hamilton's equations are integrated from (PHI1_0, J1_0) with scipy:

    dphi1/dt =  dH/dJ1 = delta + 2 alpha11 J1 + alpha12 J2
                         + m/2 G (m J1)^(m/2 - 1) (J2 - n J1)^(n/2 - 1)
                           [m J2 - n (m + n) J1] cos(phi1)
    dJ1/dt   = -dH/dphi1 = G (m J1)^(m/2) (J2 - n J1)^(n/2) sin(phi1)

The initial condition must satisfy 0 < J1_0 < J2 / n. The relative drift of
H along the orbit is printed as a check of the integration.
"""

# %% Imports
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %% Parameters
try:    # figures are saved next to this script
    FIG_DIR = Path(__file__).resolve().parent
except NameError:    # cells run one by one in an interactive window
    FIG_DIR = Path.cwd()
FIG_PREFIX = '01b'

RESONANCE = (1, 2)    # (m, n)
DELTAS = (-1.0, 0.25, 1.0)   # one curve per value
ALPHA_11 = 0.5
ALPHA_12 = 0.0
G = 1.0
J2 = 1.0

PHI1_0 = 0.0          # initial condition
J1_0 = 0.05
T_MAX = 20.0          # integration time
N_OUT = 2000          # number of output times

FIG_TAG = (f'res{RESONANCE[0]}-{RESONANCE[1]}_delta{"_".join(map(str, DELTAS))}_a11{ALPHA_11}'
           f'_a12{ALPHA_12}_phi0{PHI1_0}_J0{J1_0}')


# %% Functions
def hamiltonian(phi1, j1, delta):
    m, n = RESONANCE
    return (delta * j1 + ALPHA_12 * j1 * J2 + ALPHA_11 * j1**2
            + G * (m * j1) ** (m / 2) * (J2 - n * j1) ** (n / 2) * np.cos(phi1))


def rhs(phi1, j1, delta):
    """(dphi1/dt, dJ1/dt) from Hamilton's equations."""
    m, n = RESONANCE
    dphi = (delta + 2 * ALPHA_11 * j1 + ALPHA_12 * J2
            + m / 2 * G * (m * j1) ** (m / 2 - 1) * (J2 - n * j1) ** (n / 2 - 1)
            * (m * J2 - n * (m + n) * j1) * np.cos(phi1))
    dj = G * (m * j1) ** (m / 2) * (J2 - n * j1) ** (n / 2) * np.sin(phi1)
    return dphi, dj


# %% Integration
t = np.linspace(0, T_MAX, N_OUT)
results = {}
for delta in DELTAS:
    sol = solve_ivp(lambda _, u: rhs(*u, delta), (0, T_MAX), [PHI1_0, J1_0],
                    t_eval=t, rtol=1e-10, atol=1e-12)
    phi1, j1 = sol.y
    h = hamiltonian(phi1, j1, delta)
    print(f'delta = {delta}: relative drift of H = '
          f'{np.nanmax(np.abs(h - h[0])) / abs(h[0]):.2e}')
    results[delta] = (phi1, rhs(phi1, j1, delta)[0])

# %% Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for delta, (phi1, dphi1_dt) in results.items():
    axes[0].plot(t, np.mod(phi1 + np.pi, 2 * np.pi) - np.pi, '.', ms=1.5, label=f'delta = {delta}')
    axes[1].plot(t, dphi1_dt)
axes[0].set_ylabel('phi1 in [-pi, pi)')
axes[0].set_ylim(-np.pi, np.pi)
axes[0].legend(fontsize=8, markerscale=6, loc='upper left', bbox_to_anchor=(1.01, 1))
axes[1].axhline(0, color='0.5', lw=0.8)
axes[1].set_ylabel('dphi1/dt')
axes[1].set_xlabel('t')
fig.suptitle(f'(m, n) = {RESONANCE}, alpha11 = {ALPHA_11}, '
             f'alpha12 = {ALPHA_12}, phi1(0) = {PHI1_0}, J1(0) = {J1_0}', fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / f'{FIG_PREFIX}_phi1_dynamics_{FIG_TAG}.png', dpi=200)
plt.show()

# %%
