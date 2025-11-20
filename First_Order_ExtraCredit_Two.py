import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Import what we need from your existing 1st-order module
from First_Order import (
    free_carriers_first_order,
    kB, q, Tref,
    Si, SiC, InAs
)


# -------------------------
# Mobility parameters
# -------------------------
class MobilityParams:
    def __init__(self,
                 mu_ph0_n, alpha_ph_n, mu_imp0_n, Nref_n,
                 mu_ph0_p, alpha_ph_p, mu_imp0_p, Nref_p):
        self.mu_ph0_n  = mu_ph0_n
        self.alpha_ph_n = alpha_ph_n
        self.mu_imp0_n = mu_imp0_n
        self.Nref_n    = Nref_n

        self.mu_ph0_p  = mu_ph0_p
        self.alpha_ph_p = alpha_ph_p
        self.mu_imp0_p = mu_imp0_p
        self.Nref_p    = Nref_p


# Ballpark parameters (cm^2/Vs, cm^-3)
MOBILITY_MODELS = {
    "Silicon": MobilityParams(
        mu_ph0_n=1350.0, alpha_ph_n=1.5, mu_imp0_n=1.0e5, Nref_n=1.0e17,
        mu_ph0_p= 480.0, alpha_ph_p=1.5, mu_imp0_p=5.0e4, Nref_p=1.0e17,
    ),
    "4H-SiC": MobilityParams(
        mu_ph0_n=1000.0, alpha_ph_n=1.5, mu_imp0_n=5.0e4, Nref_n=1.0e18,
        mu_ph0_p= 115.0, alpha_ph_p=1.5, mu_imp0_p=1.0e4, Nref_p=5.0e18,
    ),
    "InAs": MobilityParams(
        mu_ph0_n=30000.0, alpha_ph_n=1.5, mu_imp0_n=2.0e5, Nref_n=5.0e16,
        mu_ph0_p=  500.0, alpha_ph_p=1.5, mu_imp0_p=5.0e4, Nref_p=1.0e18,
    ),
}


# -------------------------
# Mobility helper functions
# -------------------------
def mu_phonon(T, mu0, alpha):
    """
    Phonon-limited mobility ~ T^{-alpha}
    """
    T = np.asarray(T, dtype=float)
    return mu0 * (T / Tref) ** (-alpha)


def mu_ionized_impurity(T, N, mu0_imp, Nref):
    """
    Ionized-impurity-limited mobility: increases with T^{3/2} and
    decreases as total ionized dopants N increase.
    """
    T = np.asarray(T, dtype=float)
    N = np.asarray(N, dtype=float)
    return mu0_imp * (T / Tref) ** (1.5) / (1.0 + N / Nref)


def combine_mobility(mu_ph, mu_imp):
    """
    Matthiessen's rule: 1/mu_tot = 1/mu_ph + 1/mu_imp
    """
    mu_ph  = np.asarray(mu_ph, dtype=float)
    mu_imp = np.asarray(mu_imp, dtype=float)
    return 1.0 / (1.0 / np.maximum(mu_ph, 1e-30) +
                  1.0 / np.maximum(mu_imp, 1e-30))


def electron_mobility(mat, T, N_scatter):
    params = MOBILITY_MODELS[mat.name]
    mu_ph  = mu_phonon(T, params.mu_ph0_n, params.alpha_ph_n)
    mu_imp = mu_ionized_impurity(T, N_scatter,
                                 params.mu_imp0_n, params.Nref_n)
    return combine_mobility(mu_ph, mu_imp)


def hole_mobility(mat, T, N_scatter):
    params = MOBILITY_MODELS[mat.name]
    mu_ph  = mu_phonon(T, params.mu_ph0_p, params.alpha_ph_p)
    mu_imp = mu_ionized_impurity(T, N_scatter,
                                 params.mu_imp0_p, params.Nref_p)
    return combine_mobility(mu_ph, mu_imp)


# -------------------------
# Conductivity with 1st-order carriers
# -------------------------
def conductivity_first_order(mat, T, NA, ND):
    """
    σ(T, NA, ND) = q [ n μ_n(T,N) + p μ_p(T,N) ]
    where n,p come from your 1st-order model,
    and μ_n, μ_p come from a phonon+impurity mobility model.
    """
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    # 1) free carriers from your 1st-order solution
    n, p = free_carriers_first_order(mat, T, NA, ND)

    # 2) total ionized impurity concentration (for scattering)
    N_scatter = np.maximum(NA + ND, 1e-5)

    # 3) mobilities
    mu_n = electron_mobility(mat, T, N_scatter)
    mu_p = hole_mobility(mat, T, N_scatter)

    # 4) conductivity in S/cm
    sigma = q * (n * mu_n + p * mu_p)

    return sigma, mu_n, mu_p, n, p


# -------------------------
# Contour plots: σ vs ND & T (n-type) for A/B/C
# -------------------------

def plot_sigma_map_n_type(mat, ND_vals, T_vals, NA_fixed=1e14,
                      fname="sigma_n_map.png", title_prefix=""):

    ND_grid, T_grid = np.meshgrid(ND_vals, T_vals, indexing="ij")
    NA_grid = np.full_like(ND_grid, NA_fixed)

    # your 1st-order p(T, NA, ND)
    n, p = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # mobility model (whatever you already had)
    mu_n = electron_mobility(mat, T_grid, ND_grid)   # cm^2 / (V·s)

    # σ in S/cm
    sigma = q * n * mu_n * 1e-4   # (q [C]) * (cm^-3) * (cm^2/Vs) → A/(V·cm) = S/cm

    # ----- NEW: clip for better contrast -----
    # Ignore crazy huge values and very tiny ones
    vmin = 1e-2    # S/cm
    vmax = 1e4     # S/cm
    sigma_clipped = np.clip(sigma, vmin, vmax)

    plt.figure(figsize=(9, 6))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)

    cs = plt.contourf(ND_grid, T_grid, sigma_clipped,
                      levels=levels,
                      norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="viridis")

    # ----- NEW: add contour lines to show “borders” -----
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)
    plt.contour(ND_grid, T_grid, sigma_clipped,
                levels=contour_levels,
                colors="k", linewidths=0.4)

    plt.xscale("log")
    plt.xlabel(r"$N_D$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Conductivity $\sigma$ (n-type, 1st order carriers)")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\sigma$ (S/cm)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()




# If you also want p-type maps (σ vs NA & T), you can make a similar function:

def plot_sigma_map_p_type(mat, NA_vals, T_vals, ND_fixed=1e14,
                      fname="sigma_p_map.png", title_prefix=""):

    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij")
    ND_grid = np.full_like(NA_grid, ND_fixed)

    # your 1st-order p(T, NA, ND)
    _, p = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # mobility model (whatever you already had)
    mu_p = hole_mobility(mat, T_grid, NA_grid)   # cm^2 / (V·s)

    # σ in S/cm
    sigma = q * p * mu_p * 1e-4   # (q [C]) * (cm^-3) * (cm^2/Vs) → A/(V·cm) = S/cm

    # ----- NEW: clip for better contrast -----
    # Ignore crazy huge values and very tiny ones
    vmin = 1e-2    # S/cm
    vmax = 1e4     # S/cm
    sigma_clipped = np.clip(sigma, vmin, vmax)

    plt.figure(figsize=(9, 6))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)

    cs = plt.contourf(NA_grid, T_grid, sigma_clipped,
                      levels=levels,
                      norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="viridis")

    # ----- NEW: add contour lines to show “borders” -----
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)
    plt.contour(NA_grid, T_grid, sigma_clipped,
                levels=contour_levels,
                colors="k", linewidths=0.4)

    plt.xscale("log")
    plt.xlabel(r"$N_A$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Conductivity $\sigma$ (p-type, 1st order carriers)")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\sigma$ (S/cm)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()









# -------------------------
# Run maps for Conditions A, B, C
# -------------------------
if __name__ == "__main__":
    # Common ND axis
    ND_vals = np.logspace(14, 20, 121)
    NA_vals = np.logspace(14, 20, 121)

    # Condition A: Silicon
    T_vals_Si = np.linspace(200, 900, 181)
    plot_sigma_map_n_type(Si,
                          ND_vals=ND_vals,
                          T_vals=T_vals_Si,
                          NA_fixed=1e14,
                          title_prefix="Silicon:",
                          fname="fig_Si_sigma_n_type.png")
    plot_sigma_map_p_type(Si,
                          NA_vals=NA_vals,
                          T_vals=T_vals_Si,
                          ND_fixed=1e14,
                          title_prefix="Silicon:",
                          fname="fig_Si_sigma_p_type.png")

    # Condition B: 4H-SiC – higher T window
    T_vals_SiC = np.linspace(200, 1500, 181)
    plot_sigma_map_n_type(SiC,
                          ND_vals=ND_vals,
                          T_vals=T_vals_SiC,
                          NA_fixed=1e14,
                          title_prefix="4H-SiC:",
                          fname="fig_SiC_sigma_n_type.png")
    plot_sigma_map_p_type(SiC,
                          NA_vals=NA_vals,
                          T_vals=T_vals_SiC,
                          ND_fixed=1e14,
                          title_prefix="4H-SiC:",
                          fname="fig_SiC_sigma_p_type.png")

    # Condition C: InAs – narrower bandgap, lower T window
    T_vals_InAs = np.linspace(200, 700, 161)
    plot_sigma_map_n_type(InAs,
                          ND_vals=ND_vals,
                          T_vals=T_vals_InAs,
                          NA_fixed=1e14,
                          title_prefix="InAs:",
                          fname="fig_InAs_sigma_n_type.png")
    plot_sigma_map_p_type(InAs,
                          NA_vals=NA_vals,
                          T_vals=T_vals_InAs,
                          ND_fixed=1e14,
                          title_prefix="InAs:",
                          fname="fig_InAs_sigma_p_type.png")
