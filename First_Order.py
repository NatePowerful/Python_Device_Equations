import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass
# import Zero_Order   # <- keep commented out unless you actually have this module

#need constraint if user inputs  unrealistic data

# -------------------------
# Universal constants
# -------------------------
q   = 1.602176634e-19        # C
kB  = 1.380649e-23           # J/K, boltzmann const.
h   = 6.62607015e-34         # J*s, planck const.
m0  = 9.1093837015e-31       # kg free electron rest mass
Tref = 300.0                 # K ref. temperature

# SI → cm^-3 conversion: results from the DOS formula are in m^-3; multiply by 1e-6 to get cm^-3
M3_TO_CM3 = 1e-6

# -------------------------
# Basic input validation (helps prevent unrealistic inputs)
# -------------------------
def validate_inputs(T, NA, ND):
    if np.any(T < 0):
        raise ValueError("Temperature must be ≥ 0 K.")
    if np.any(NA < 0) or np.any(ND < 0):
        raise ValueError("Dopant concentrations must be ≥ 0 cm^-3.")
    if np.any(NA > 1e21) or np.any(ND > 1e21):
        print("Warning: dopings above ~1e21 cm^-3 are nonphysical for most semiconductors.")

# -------------------------
# DOS helpers (from first principles, includes π and h explicitly)
# m_eff is DOS effective mass in units of m0 (electron rest mass)
# -------------------------
def Nc_physical(T, m_eff):
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)  # Added safety guard to avoid T=0
    return (2.0 * ((2.0 * np.pi * (m_eff * m0) * kB * T) / (h**2))**1.5) * M3_TO_CM3

def Nv_physical(T, m_eff):
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return (2.0 * ((2.0 * np.pi * (m_eff * m0) * kB * T) / (h**2))**1.5) * M3_TO_CM3

# -------------------------
# Material model (0th order)
# -------------------------
@dataclass
class Material0th:
    name: str
    mstar_n: float              # DOS effective mass for electrons, in units of m0, assuming given
    mstar_p: float              # DOS effective mass for holes, in units of m0, assuming given
    E_gap: float                #given energy bandgap of material in eV, assuming negligible change for temp. window (for now...)
 
    def Nc(self, T):
        return Nc_physical(T, self.mstar_n)

    def Nv(self, T):
        return Nv_physical(T, self.mstar_p)

    def ni(self, T):
        Eg_J = self.E_gap * q                    # eV -> J
        Nc = self.Nc(T)
        Nv = self.Nv(T)
        return np.sqrt(Nc * Nv) * np.exp(-Eg_J / (2.0 * kB * T))  # cm^-3

# -------------------------
# Example: Silicon (put YOUR class parameters here)
# mstar_n, mstar_p are DOS masses (not transport masses). The values below are common approximations;
# replace with the exact values your instructor/text provides.
# -------------------------
Si = Material0th(
    name="Silicon",
    mstar_n=1.08,     # DOS electron mass (units of m0) – example
    mstar_p=0.81,     # DOS hole mass (units of m0) – example
    E_gap = 1.1
)

SiC = Material0th(
    name="4H-SiC",
    mstar_n=0.42,   # DOS electron mass / m0  (placeholder; use your class numbers if given)
    mstar_p=0.66,   # DOS hole mass / m0      (placeholder)
    E_gap=2.2      # eV (approx at 300K per slide list)
)

InAs = Material0th(
    name="InAs",
    mstar_n=0.026,  # DOS electron mass / m0  (light electrons)
    mstar_p=0.41,   # DOS hole mass / m0      (placeholder)
    E_gap=0.354     # eV (approx at 300K)
)

# -------------------------
# 1st-order equations: Free carriers (complete ionization)
# -------------------------
def free_carriers_first_order(mat, T, NA, ND):
    """
    Compute 1st-order free carriers (complete ionization) for given material.

    Parameters
    ----------
    mat : object with ni(T) -> intrinsic concentration in cm^-3
    T   : K
    NA  : cm^-3  (acceptor density)
    ND  : cm^-3  (donor density)

    Returns
    -------
    n, p : cm^-3
    """
    # Added safety clamps
    T  = np.maximum(np.asarray(T,  dtype=float), 1e-12)
    NA = np.maximum(np.asarray(NA, dtype=float), 0.0)
    ND = np.maximum(np.asarray(ND, dtype=float), 0.0)

    # Optional validation step
    # validate_inputs(T, NA, ND)

    ni = mat.ni(T)                                 # cm^-3
    dN = ND - NA
    # Guard against tiny numerical negatives under the square root:
    root = np.sqrt(np.maximum((dN/2.0)**2 + ni**2, 0.0))
    n = root + dN/2.0
    # Avoid division by zero:
    n = np.maximum(n, 1e-30)
    p = (ni**2) / n
    return n, p

# ---------- Plot functions ----------

#Here, we are leaving NA fixed while ND sweeps
def plot_n_map_vs_ND_T(mat,
                       ND_vals=np.logspace(14, 19, 121),
                       T_vals=np.linspace(200, 900, 181),
                       NA_fixed=1e14,
                       fname="fig_free_n_map.png",
                       title_prefix="",
                       use_log_color=True):

    # Grids
    ND_grid, T_grid = np.meshgrid(ND_vals, T_vals, indexing="ij")
    NA_grid = np.full_like(ND_grid, NA_fixed)

    # 1st-order carriers
    n, _ = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # Degeneracy warning
    NcT = mat.Nc(T_grid)
    deg_mask = (n / np.maximum(NcT, 1e-300)) >= np.exp(-3.0)
    if np.any(deg_mask):
        print(f"Warning [{mat.name} n-map]: region includes degenerate electrons (EF near/inside Ec). "
              "1st-order Boltzmann may be inaccurate there.")

    # ---------- Improved visibility ----------
    # Choose a physically interesting window and clip to it
    vmin = 1e8   # cm^-3
    vmax = 1e20  # cm^-3
    n_plot = np.clip(n, vmin, vmax)

    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)

    plt.figure(figsize=(10, 6))
    if use_log_color:
        cs = plt.contourf(ND_grid, T_grid, n_plot,
                          levels=levels,
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap="plasma")  # higher contrast
    else:
        cs = plt.contourf(ND_grid, T_grid, n_plot, levels=30, cmap="plasma")

    # Add contour lines
    plt.contour(ND_grid, T_grid, n_plot,
                levels=np.logspace(8, 20, 7),
                colors="black", linewidths=0.5)

    # Degenerate region hatching
    plt.contour(ND_grid, T_grid, deg_mask.astype(float),
                levels=[0.5], colors="k", linewidths=1.0)
    plt.contourf(ND_grid, T_grid, deg_mask.astype(float),
                 levels=[0.5, 1.1], hatches=["///"], colors="none", alpha=0)

    plt.xscale("log")
    plt.xlabel(r"$N_D$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Free Electron Concentration n (1st order)")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"n (cm$^{-3}$)")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

#Here, we are leaving ND fixed while NA sweeps
def plot_p_map_vs_NA_T(mat,
                       NA_vals=np.logspace(14, 19, 121),
                       T_vals=np.linspace(200, 900, 181),
                       ND_fixed=1e14,
                       fname="fig_free_p_map.png",
                       title_prefix="",
                       use_log_color=True):

    # Grids
    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij")
    ND_grid = np.full_like(NA_grid, ND_fixed)

    # 1st-order carriers
    _, p = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # Degeneracy warning
    NvT = mat.Nv(T_grid)
    deg_mask = (p / np.maximum(NvT, 1e-300)) >= np.exp(-3.0)
    if np.any(deg_mask):
        print(f"Warning [{mat.name} p-map]: region includes degenerate holes (EF near/inside Ev). "
              "1st-order Boltzmann may be inaccurate there.")

    # ---------- Improved visibility ----------
    vmin = 1e8   # cm^-3
    vmax = 1e20  # cm^-3
    p_plot = np.clip(p, vmin, vmax)

    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)

    plt.figure(figsize=(10, 6))
    if use_log_color:
        cs = plt.contourf(NA_grid, T_grid, p_plot,
                          levels=levels,
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap="plasma")
    else:
        cs = plt.contourf(NA_grid, T_grid, p_plot, levels=30, cmap="plasma")

    # Contour lines
    plt.contour(NA_grid, T_grid, p_plot,
                levels=np.logspace(8, 20, 7),
                colors="black", linewidths=0.5)

    # Degenerate region hatching
    plt.contour(NA_grid, T_grid, deg_mask.astype(float),
                levels=[0.5], colors="k", linewidths=1.0)
    plt.contourf(NA_grid, T_grid, deg_mask.astype(float),
                 levels=[0.5, 1.1], hatches=["///"], colors="none", alpha=0)

    plt.xscale("log")
    plt.xlabel(r"$N_A$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Free Hole Concentration p (1st order)")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"p (cm$^{-3}$)")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# -------------------------
# Run 1st-order plots for all conditions (A/B/C)
# -------------------------
if __name__ == "__main__":
    # Condition A: Silicon
    plot_n_map_vs_ND_T(Si,
                       title_prefix="Silicon:",
                       fname="fig_Si_free_n_map.png")
    plot_p_map_vs_NA_T(Si,
                       title_prefix="Silicon:",
                       fname="fig_Si_free_p_map.png")

    # Condition B: 4H-SiC — higher temperature window
    plot_n_map_vs_ND_T(SiC,
                       T_vals=np.linspace(200, 1500, 181),
                       title_prefix="4H-SiC:",
                       fname="fig_SiC_free_n_map.png")
    plot_p_map_vs_NA_T(SiC,
                       T_vals=np.linspace(200, 1500, 181),
                       title_prefix="4H-SiC:",
                       fname="fig_SiC_free_p_map.png")

    # Condition C: InAs — narrower bandgap; slightly lower temperature window
    plot_n_map_vs_ND_T(InAs,
                       T_vals=np.linspace(200, 700, 161),
                       title_prefix="InAs:",
                       fname="fig_InAs_free_n_map.png")
    plot_p_map_vs_NA_T(InAs,
                       T_vals=np.linspace(200, 700, 161),
                       title_prefix="InAs:",
                       fname="fig_InAs_free_p_map.png")

    # show them interactively (optional if you just save figures)
    plt.show()
