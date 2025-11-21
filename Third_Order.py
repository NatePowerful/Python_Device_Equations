# actually the 3rd order
# Third_Order.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------------------
# Import 0th-order material model and constants
# -------------------------
import Zero_Order as Z0
# We expect Zero_Order.py to define:
#   q, kB, Material0th, Si, SiC, InAs
q   = Z0.q
kB  = Z0.kB
Si  = Z0.Si
SiC = Z0.SiC
InAs = Z0.InAs

# Permittivity of free space in F/cm (needed for depletion width, needed for 3rd order)
eps0 = 8.8541878128e-14  # F/cm

# Material-specific permittivity and breakdown fields for 3rd order
# (You can tweak these if your instructor gives specific numbers.)
MAT_3RD_PARAMS = {
    "Silicon": {
        "eps_rel": 11.7,   # relative permittivity
        "Ecrit":   3.0e5,  # critical field [V/cm]
    },
    "4H-SiC": {
        "eps_rel": 9.7,
        "Ecrit":   3.0e6,
    },
    "InAs": {
        "eps_rel": 15.15,
        "Ecrit":   2.0e5,
    },
}

def get_eps_rel(mat):
    """Return relative permittivity for the given material (by name)."""
    params = MAT_3RD_PARAMS.get(mat.name)
    if params is None:
        # default to Silicon-like if not found
        return 11.7
    return params["eps_rel"]

def get_Ecrit(mat):
    """Return critical breakdown field (V/cm) for the given material."""
    params = MAT_3RD_PARAMS.get(mat.name)
    if params is None:
        #default if not provided
        return 3.0e5
    return params["Ecrit"]

# -------------------------
# 3rd-order main equations:
#   - built-in potential φ_bi
#   - depletion width W
#   - E_max gives peak E-field
# -------------------------

def builtin_potential(mat, T, NA, ND):
    """
    Built-in voltage (phi_bi) for an abrupt pn junction.

    Parameters
    ----------
    mat : Material0th (needs ni(T))
    T   : K
    NA  : cm^-3  (acceptor density)
    ND  : cm^-3  (donor density)

    Returns
    -------
    phi_bi : V
    """
    T  = np.maximum(np.asarray(T,  dtype=float), 1e-12)
    NA = np.maximum(np.asarray(NA, dtype=float), 1e-30)
    ND = np.maximum(np.asarray(ND, dtype=float), 1e-30)

    ni  = mat.ni(T)                         # units in cm^-3
    ni2 = np.maximum(ni**2, 1e-60)
    prod = np.maximum(NA * ND, 1e-60)

    kT_over_q = (kB * T) / q                # units involts
    phi_bi = kT_over_q * np.log(prod / ni2) # units V
    # guard against tiny negative values in root from rounding
    phi_bi = np.maximum(phi_bi, 0.0)
    return phi_bi


def depletion_width(mat, T, NA, ND):
    """
    Total depletion width W for an abrupt pn junction.

    Formula (no external bias):
        W = sqrt( (2 * eps_s / q) * (phi_bi) * (1/NA + 1/ND) )

    Parameters
    ----------
    mat : Material0th
    T   : K
    NA  : cm^-3
    ND  : cm^-3

    Returns
    -------
    W : cm
    """
    #safeguards against unrealistic values/divsion by zero
    T  = np.maximum(np.asarray(T,  dtype=float), 1e-12)
    NA = np.maximum(np.asarray(NA, dtype=float), 1e-30)
    ND = np.maximum(np.asarray(ND, dtype=float), 1e-30)

    phi_bi = builtin_potential(mat, T, NA, ND)
    eps_rel = get_eps_rel(mat)
    eps_s = eps_rel * eps0           # units in F/cm

    factor = (1.0 / NA) + (1.0 / ND)
    W = np.sqrt(2.0 * eps_s * phi_bi / q * factor)  # units in cm
    return W


def peak_field(mat, T, NA, ND):
    """
    Approximate maximum electric field at the junction (no applied bias).

    A simple estimate:
        E_max ≈ 2 * phi_bi / W     [V/cm]

    Returns
    -------
    Emax : V/cm
    """
    phi_bi = builtin_potential(mat, T, NA, ND)
    W = depletion_width(mat, T, NA, ND)
    W = np.maximum(W, 1e-20)
    return 2.0 * phi_bi / W

# 3rd-order plot functions

def plot_phi_bi_map_vs_NA_T(mat,
                            NA_vals=np.logspace(14, 19, 121),
                            T_vals=np.linspace(200, 900, 181),
                            ND_fixed=1e16,
                            fname="fig_phi_bi_map.png",
                            title_prefix="",
                            use_log_color=False):
    """
    Contour map of built-in potential vs NA and T (ND fixed).

    Also overlays a hatched region where E_max >= critical field (breakdown threshold).
    """
    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij")
    ND_grid = np.full_like(NA_grid, ND_fixed)

    phi_bi = builtin_potential(mat, T_grid, NA_grid, ND_grid)  # V
    Emax   = peak_field(mat, T_grid, NA_grid, ND_grid)        # V/cm
    Ecrit  = get_Ecrit(mat)

    bd_mask = (Emax >= Ecrit)  # breakdown region

    plt.figure()
    if use_log_color:
        vmin = max(np.nanmax(phi_bi) * 1e-3, 1e-3)  # avoid zero
        vmax = max(np.nanmax(phi_bi), vmin * 1e2)
        cs = plt.contourf(NA_grid, T_grid, phi_bi, levels=30,
                          norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        cs = plt.contourf(NA_grid, T_grid, phi_bi, levels=30)

    # Breakdown region: outline + hatching
    plt.contour(NA_grid, T_grid, bd_mask.astype(float),
                levels=[0.5], colors="k", linewidths=1.0)
    plt.contourf(NA_grid, T_grid, bd_mask.astype(float),
                 levels=[0.5, 1.1], hatches=["xxx"], colors="none", alpha=0)

    plt.xscale("log")
    plt.xlabel(r"$N_A$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Built-in Potential $\\varphi_{{bi}}$ (3rd order)")
    cbar = plt.colorbar(cs); cbar.set_label(r"$\varphi_{bi}$ (V)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_W_map_vs_NA_T(mat,
                       NA_vals=np.logspace(14, 19, 121),
                       T_vals=np.linspace(200, 900, 181),
                       ND_fixed=1e16,
                       fname="fig_W_map.png",
                       title_prefix="",
                       use_log_color=True):
    """
    Contour map of depletion width W vs NA and T (ND fixed).

    Also overlays a hatched region where E_max >= critical field.
    """
    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij")
    ND_grid = np.full_like(NA_grid, ND_fixed)

    W = depletion_width(mat, T_grid, NA_grid, ND_grid)  #units in cm
    Emax   = peak_field(mat, T_grid, NA_grid, ND_grid)  #units in V/cm
    Ecrit  = get_Ecrit(mat)
    bd_mask = (Emax >= Ecrit)

    plt.figure()
    if use_log_color:
        vmin = max(np.nanmax(W) * 1e-3, 1e-7)   # cm; floor of roughly 1 nm
        vmax = max(np.nanmax(W), vmin * 1e2)
        cs = plt.contourf(NA_grid, T_grid, W, levels=30,
                          norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        cs = plt.contourf(NA_grid, T_grid, W, levels=30)

    # Breakdown region: outline + hatching
    plt.contour(NA_grid, T_grid, bd_mask.astype(float),
                levels=[0.5], colors="k", linewidths=1.0)
    plt.contourf(NA_grid, T_grid, bd_mask.astype(float),
                 levels=[0.5, 1.1], hatches=["xxx"], colors="none", alpha=0)

    #plot logic
    plt.xscale("log")
    plt.xlabel(r"$N_A$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Depletion Width $W$ (3rd order)")
    cbar = plt.colorbar(cs); cbar.set_label(r"$W$ (cm)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

#plots for all conditions
if __name__ == "__main__":
    # Silicon: moderate temps relative to others
    plot_phi_bi_map_vs_NA_T(Si,
                            ND_fixed=1e16,
                            title_prefix="Silicon:",
                            fname="fig_Si_phi_bi_map.png")
    plot_W_map_vs_NA_T(Si,
                       ND_fixed=1e16,
                       title_prefix="Silicon:",
                       fname="fig_Si_W_map.png")

    # 4H-SiC: higher temperature window (higher temp)
    plot_phi_bi_map_vs_NA_T(SiC,
                            T_vals=np.linspace(200, 1500, 181),
                            ND_fixed=1e16,
                            title_prefix="4H-SiC:",
                            fname="fig_SiC_phi_bi_map.png")
    plot_W_map_vs_NA_T(SiC,
                       T_vals=np.linspace(200, 1500, 181),
                       ND_fixed=1e16,
                       title_prefix="4H-SiC:",
                       fname="fig_SiC_W_map.png")

    # InAs: narrower gap; keep temps lower
    plot_phi_bi_map_vs_NA_T(InAs,
                            T_vals=np.linspace(200, 700, 161),
                            ND_fixed=1e16,
                            title_prefix="InAs:",
                            fname="fig_InAs_phi_bi_map.png")
    plot_W_map_vs_NA_T(InAs,
                       T_vals=np.linspace(200, 700, 161),
                       ND_fixed=1e16,
                       title_prefix="InAs:",
                       fname="fig_InAs_W_map.png")

    plt.show()
