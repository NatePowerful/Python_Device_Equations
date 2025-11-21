import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Import from 1st-order module
from First_Order import (
    free_carriers_first_order,
    kB, q, Tref,
    Si, SiC, InAs
)



# Mobility parameters
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



# Mobility helper functions
def mu_phonon(T, mu0, alpha):
    """
    Phonon-limited mobility ~ T^{-alpha}
    """
    T = np.asarray(T, dtype=float)
    return mu0 * (T / Tref) ** (-alpha)


def mu_ionized_impurity(T, N, mu0_imp, Nref):
    """
    "Ionized-impurity-limited" mobility: increases as with T^{3/2} and
    decreases as total ionized dopants N increase.
    """
    T = np.asarray(T, dtype=float)
    N = np.asarray(N, dtype=float)
    return mu0_imp * (T / Tref) ** (1.5) / (1.0 + N / Nref)


def combine_mobility(mu_ph, mu_imp):
    """
    "Matthiessen's rule": 1/mu_tot = 1/mu_ph + 1/mu_imp
    """
    mu_ph  = np.asarray(mu_ph, dtype=float)
    mu_imp = np.asarray(mu_imp, dtype=float)
    return 1.0 / (1.0 / np.maximum(mu_ph, 1e-30) +
                  1.0 / np.maximum(mu_imp, 1e-30))


def electron_mobility(mat, T, N_scatter):
    """
    electron mobility, should have higher mobility due to parameters chosen
    """
    params = MOBILITY_MODELS[mat.name] #indexed at material name
    mu_ph  = mu_phonon(T, params.mu_ph0_n, params.alpha_ph_n) #passing in params from material
    mu_imp = mu_ionized_impurity(T, N_scatter,
                                 params.mu_imp0_n, params.Nref_n)
    return combine_mobility(mu_ph, mu_imp) #return vals


def hole_mobility(mat, T, N_scatter):
    '''
    same as above, just now with holes instead of electrons, now should have lower mobility
    '''
    params = MOBILITY_MODELS[mat.name]
    mu_ph  = mu_phonon(T, params.mu_ph0_p, params.alpha_ph_p)
    mu_imp = mu_ionized_impurity(T, N_scatter,
                                 params.mu_imp0_p, params.Nref_p)
    return combine_mobility(mu_ph, mu_imp)



# Conductivity w/ 1st-order carriers
def conductivity_first_order(mat, T, NA, ND):
    """
    σ(T, NA, ND) = q [ n μ_n(T,N) + p μ_p(T,N) ]
    where n,p come 1st order,
    and μ_n, μ_p come from a phonon+impurity mobility model.
    """
    T  = np.asarray(T,  dtype=float) #define as arrays for T, NA, and ND, data type is float to hold decimal values
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    # free carriers from your 1st-order
    n, p = free_carriers_first_order(mat, T, NA, ND)

    # total ionized impurity concentration (scattering)
    N_scatter = np.maximum(NA + ND, 1e-5)

    # mobilities
    mu_n = electron_mobility(mat, T, N_scatter)
    mu_p = hole_mobility(mat, T, N_scatter)

    # conductivity in S/cm
    sigma = q * (n * mu_n + p * mu_p) #number of electrons, number of holes, each multiplied the mobility

    return sigma, mu_n, mu_p, n, p



# Contour plots: σ vs ND & T (n-type) for different material conditions

def plot_sigma_map_n_type(mat, ND_vals, T_vals, NA_fixed=1e14,
                      fname="sigma_n_map.png", title_prefix=""):
    
    #since n-type, ND vals fluctuate, NA is fixed

    ND_grid, T_grid = np.meshgrid(ND_vals, T_vals, indexing="ij")
    NA_grid = np.full_like(ND_grid, NA_fixed)

    #1st-order p(T, NA, ND)
    n, p = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # mobility model 
    mu_n = electron_mobility(mat, T_grid, ND_grid)   # cm^2 / (V·s)

    # σ in S/cm
    sigma = q * n * mu_n * 1e-4   # (q [C]) * (cm^-3) * (cm^2/Vs) → A/(V·cm) = S/cm


    # Ignore unrealistic values (large and small) with clip
    vmin = 1e-2    # S/cm
    vmax = 1e4     # S/cm
    sigma_clipped = np.clip(sigma, vmin, vmax)

    plt.figure(figsize=(9, 6)) #set plot dimensions
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40) #setting levels by logspace, 40 values between Vmin and Vmax
     
    #variable set for contour plotting
    cs = plt.contourf(ND_grid, T_grid, sigma_clipped,
                      levels=levels,
                      norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="viridis")

    #contour lines to show borders
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)
    plt.contour(ND_grid, T_grid, sigma_clipped,
                levels=contour_levels,
                colors="k", linewidths=0.4)
    #plot logic
    plt.xscale("log")
    plt.xlabel(r"$N_D$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Conductivity $\sigma$ (n-type, 1st order carriers)")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$\sigma$ (S/cm)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()




# similar to above, just p-type now

def plot_sigma_map_p_type(mat, NA_vals, T_vals, ND_fixed=1e14,
                      fname="sigma_p_map.png", title_prefix=""):

    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij")
    ND_grid = np.full_like(NA_grid, ND_fixed) #now NA fluctuates, ND is fixed

    # 1st-order p(T, NA, ND)
    _, p = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # mobility model
    mu_p = hole_mobility(mat, T_grid, NA_grid)   # cm^2 / (V·s)

    # σ in S/cm
    sigma = q * p * mu_p * 1e-4   # (q [C]) * (cm^-3) * (cm^2/Vs) → A/(V·cm) = S/cm

    #clipping like before 
    vmin = 1e-2    # S/cm
    vmax = 1e4     # S/cm
    sigma_clipped = np.clip(sigma, vmin, vmax)

    plt.figure(figsize=(9, 6))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)

    cs = plt.contourf(NA_grid, T_grid, sigma_clipped,
                      levels=levels,
                      norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="viridis")

    # contour lines
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)
    plt.contour(NA_grid, T_grid, sigma_clipped,
                levels=contour_levels,
                colors="k", linewidths=0.4)
    
    #plot logic
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
    ND_vals = np.logspace(14, 20, 121) #ND and NA set for each plot type
    NA_vals = np.logspace(14, 20, 121)
    T_vals_Si   = np.linspace(150, 900, 300) #temp values set for each material
    T_vals_SiC  = np.linspace(200, 1500, 300)
    T_vals_InAs = np.linspace(100, 700, 300)


    #Silicon
    plot_sigma_map_n_type(Si,
                          ND_vals=ND_vals,
                          T_vals=T_vals_Si,
                          NA_fixed=1e14,
                          title_prefix="Silicon:",
                          fname="fig_Si_sigma_n_type_hightemp.png")
    plot_sigma_map_p_type(Si,
                          NA_vals=NA_vals,
                          T_vals=T_vals_Si,
                          ND_fixed=1e14,
                          title_prefix="Silicon:",
                          fname="fig_Si_sigma_p_type_hightemp.png")

    # 4H-SiC – higher T window from higher bandgap
    plot_sigma_map_n_type(SiC,
                          ND_vals=ND_vals,
                          T_vals=T_vals_SiC,
                          NA_fixed=1e14,
                          title_prefix="4H-SiC:",
                          fname="fig_SiC_sigma_n_type_hightemp.png")
    plot_sigma_map_p_type(SiC,
                          NA_vals=NA_vals,
                          T_vals=T_vals_SiC,
                          ND_fixed=1e14,
                          title_prefix="4H-SiC:",
                          fname="fig_SiC_sigma_p_type_hightemp.png")

    # InAs – narrower bandgap, lower T window
    plot_sigma_map_n_type(InAs,
                          ND_vals=ND_vals,
                          T_vals=T_vals_InAs,
                          NA_fixed=1e14,
                          title_prefix="InAs:",
                          fname="fig_InAs_sigma_n_type_hightemp.png")
    plot_sigma_map_p_type(InAs, 
                          NA_vals=NA_vals,
                          T_vals=T_vals_InAs,
                          ND_fixed=1e14,
                          title_prefix="InAs:",
                          fname="fig_InAs_sigma_p_type_hightemp.png")
