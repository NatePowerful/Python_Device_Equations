# actually the 3rd order
# Third_Order.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Import 0th-order material model and constants
import Zero_Order as Z0

# q, kB, Material0th, Si, SiC, InAs
q   = Z0.q
kB  = Z0.kB
Si  = Z0.Si
SiC = Z0.SiC
InAs = Z0.InAs

# Permittivity of free space in F/cm (depletion width)
eps0 = 8.8541878128e-14  # F/cm

# Material-specific permittivity and breakdown
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
    #Return relative permittivity for material (by name)
    params = MAT_3RD_PARAMS.get(mat.name)
    if params is None:
        # default = Silicon-like if not found
        return 11.7
    return params["eps_rel"]

def get_Ecrit(mat):
    """Return critical breakdown field (V/cm) for the given material."""
    params = MAT_3RD_PARAMS.get(mat.name)
    if params is None:
        #default value
        return 3.0e5
    return params["Ecrit"]




# -------------------------
# Freeze-out model for dopants (extra credit)
# -------------------------

# Same energies you used in your 1st-order extra credit
E_D_Si   = 0.045   # eV (assuming shallow donor in Si)
E_A_Si   = 0.045   # eV (assuming shallow acceptor in Si)

E_D_SiC  = 0.20    # eV (donor in 4H-SiC)
E_A_SiC  = 0.20    # eV (acceptor in 4H-SiC)

E_D_InAs = 0.006   # eV (shallow donor in InAs)
E_A_InAs = 0.03    # eV (acceptor in InAs)

gD = 0.5           # donor "degeneracy factor" (both set at 0.5)
gA = 0.5           # acceptor "degeneracy factor"


def _donor_ionization_fraction_simple(T, E_D_eV, gD_local=0.5):
    """
    Uses simple "donor ionization fraction":
        f_D = 1 / (1 + g_D * exp(E_D / kT))

    No Fermi-level shift; just a function of T and doping
    """
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    E_D_J = E_D_eV * q
    return 1.0 / (1.0 + gD_local * np.exp(E_D_J / (kB * T)))


def _acceptor_ionization_fraction_simple(T, E_A_eV, gA_local=0.5):
    """
    Simple acceptor ionization fraction:
        f_A = 1 / (1 + g_A * exp(E_A / kT))
    """
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    E_A_J = E_A_eV * q
    return 1.0 / (1.0 + gA_local * np.exp(E_A_J / (kB * T)))




def effective_dopants_freezeout(mat, T, NA, ND):
    """
    Returns effective (ionized) dopant density N_A,eff and N_D,eff
    for the given material, uses freeze-out model.
    """
    if mat is Si:
        E_D_use, E_A_use = E_D_Si,  E_A_Si
    elif mat is SiC:
        E_D_use, E_A_use = E_D_SiC, E_A_SiC
    elif mat is InAs:
        E_D_use, E_A_use = E_D_InAs, E_A_InAs
    else:
        # default if other material appears
        E_D_use = 0.05
        E_A_use = 0.05

    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    fD = _donor_ionization_fraction_simple(T, E_D_use, gD_local=gD)
    fA = _acceptor_ionization_fraction_simple(T, E_A_use, gA_local=gA)

    ND_eff = ND * fD
    NA_eff = NA * fA
    return NA_eff, ND_eff










# -------------------------
# 3rd-order core equations:
#   - built-in potential φ_bi
#   - depletion width W
#   - peak electric field E_max
# -------------------------

def builtin_potential(mat, T, NA, ND, freezeout=False):
    """
    Built-in voltage (phi_bi) for "abrupt" pn junction.

    Parameters:
    
    mat : Material0th (needs ni(T))
    T   : K
    NA  : cm^-3  (acceptor density)
    ND  : cm^-3  (donor density)

    
    Output values are:

    phi_bi : V
    """
    """
    If freezeout=True, use effective ionized dopants (N_A,eff, N_D,eff)
    instead of the total dopant densities, otherwise if freezeout=False
    """
    T  = np.maximum(np.asarray(T,  dtype=float), 1e-12)
    NA = np.maximum(np.asarray(NA, dtype=float), 1e-30)
    ND = np.maximum(np.asarray(ND, dtype=float), 1e-30)

    if freezeout:
        NA_use, ND_use = effective_dopants_freezeout(mat, T, NA, ND)
    else:
        NA_use, ND_use = NA, ND

    ni  = mat.ni(T)                        # units in cm^-3
    ni2 = np.maximum(ni**2, 1e-60)
    prod = np.maximum(NA_use * ND_use, 1e-60)

    kT_over_q = (kB * T) / q               # units in volts
    phi_bi = kT_over_q * np.log(prod / ni2)
    phi_bi = np.maximum(phi_bi, 0.0)
    return phi_bi

  

def depletion_width(mat, T, NA, ND, freezeout=False):
    """
    Depletion width (W) for an abrupt pn junction.
    If freezeout=True, use ionized dopants only.
    """
    T  = np.maximum(np.asarray(T,  dtype=float), 1e-12) #minima for T, NA, ND, set by taking maxima
    NA = np.maximum(np.asarray(NA, dtype=float), 1e-30)
    ND = np.maximum(np.asarray(ND, dtype=float), 1e-30)

    phi_bi = builtin_potential(mat, T, NA, ND, freezeout=freezeout) #phi_bi with freezeout condition


    if freezeout:
      NA_use, ND_use = effective_dopants_freezeout(mat, T, NA, ND)
    else:
      NA_use, ND_use = NA, ND


    NA_use = np.maximum(NA_use, 1e-30)  #if unreasonably low amount of NA_use or ND_use, default to 1e-30
    ND_use = np.maximum(ND_use, 1e-30)

    eps_rel = get_eps_rel(mat)
    eps_s = eps_rel * eps0  

    factor = (1.0 / NA_use) + (1.0 / ND_use)

    W = np.sqrt(2.0 * eps_s * phi_bi / q * factor)  # units in cm
    return W



def peak_field(mat, T, NA, ND, freezeout=False):
    """
    Approximates max E-field at the junction (no applied bias).
    If freezeout=True, use ionized dopants only.
    """
    phi_bi = builtin_potential(mat, T, NA, ND, freezeout=freezeout)
    W = depletion_width(mat, T, NA, ND, freezeout=freezeout)
    W = np.maximum(W, 1e-20)
    return 2.0 * phi_bi / W






# -------------------------
# 3rd-order plot functions
# -------------------------
def plot_W_freezeout_ratio_map(mat,
                               NA_vals=np.logspace(14, 19, 121),
                               T_vals=np.linspace(50, 600, 200),
                               ND_fixed=1e16,
                               fname="fig_W_freezeout_ratio_3rdOrder.png",
                               title_prefix=""):
    """
    show how much depletion width changes
    when incomplete ionization is considered (freeze-out).

        ratio = W_freeze / W_full

    At high T, ratio approx. equal to 1.
    At low T and moderate dopings, ratio > 1 (ionization incomplete).
    """
    NA_grid, T_grid = np.meshgrid(NA_vals, T_vals, indexing="ij") #grid using NA and T values
    ND_grid = np.full_like(NA_grid, ND_fixed) #ND is fixed in this case

    # Full ionization
    W_full   = depletion_width(mat, T_grid, NA_grid, ND_grid,
                               freezeout=False)
    # With freeze-out
    W_freeze = depletion_width(mat, T_grid, NA_grid, ND_grid,
                               freezeout=True)

    ratio = W_freeze / np.maximum(W_full, 1e-30) #setting lower bound of 1e-30

    plt.figure(figsize=(8, 6)) #plot size (dimensions)
    levels = np.linspace(1.0, 5.0, 33)  # show up to 5× increase, can go more but chose 5 for practicality
    cs = plt.contourf(NA_grid, T_grid, ratio, levels=levels, cmap="viridis") #plotting variable
    plt.xscale("log") #using log scale on x-axis
    plt.xlabel(r"$N_A$ (cm$^{-3}$)") #labels
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Depletion Width Ratio "
              r"$W_{\mathrm{freeze}}/W_{\mathrm{full}}$")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$W_{\mathrm{freeze}} / W_{\mathrm{full}}$")

    #contour line at ratio = 2, “strong freeze-out” boundary
    plt.contour(NA_grid, T_grid, ratio, levels=[2.0],
                colors="red", linewidths=1.0)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()






if __name__ == "__main__":
    #calling plot functions with temp values, relatively low temps
    plot_W_freezeout_ratio_map(Si,
                               T_vals=np.linspace(100, 800, 200),
                               title_prefix="Silicon:",
                               fname="fig_Si_W_freezeout_ratio_3rdOrder.png")

    plot_W_freezeout_ratio_map(SiC,
                               T_vals=np.linspace(500, 1200, 250),
                               title_prefix="4H-SiC:",
                               fname="fig_SiC_W_freezeout_ratio_3rdOrder.png")

    plot_W_freezeout_ratio_map(InAs,
                               T_vals=np.linspace(50, 500, 200),
                               title_prefix="InAs:",
                               fname="fig_InAs_W_freezeout_ratio_3rdOrder.png")
