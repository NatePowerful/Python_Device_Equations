import numpy as np
import matplotlib.pyplot as plt

# using existing first order
from First_Order import (
    free_carriers_first_order,
    kB, q, Tref,
    Si, SiC, InAs
)

#assuming Incomplete ionization
E_D_Si   = 0.045   # P in Si about 45 meV
E_D_SiC  = 0.200   # shallow donor in 4H-SiC
E_D_InAs = 0.006   # shallow donor in InAs


E_A_Si   = 0.045   # shallow in Si
E_A_SiC  = 0.200   # "rough" acceptor in 4H-SiC
E_A_InAs = 0.03   # shallow acceptor in InAs

# degeneracy factors (typical values, assumed)
gD = 0.5
gA = 0.5



def donor_ionization_fraction_with_Fermi(mat, T, NA, ND,
                                         E_D_eV=0.045,
                                         gD_local=0.5):
    """
    Donor ionization fraction f_D = N_D^+ / N_D including
    the Fermi-level shift due to doping.

    Uses 1st-order solution to estimate n, then:
        n ≈ N_c exp(-(E_c - E_F)/kT)
        => E_F - E_D = kT ln(n/N_c) - E_D

    and
        N_D^+ = N_D / (1 + g_D exp((E_F - E_D)/kT))
              = N_D / (1 + g_D * (n/N_c) * exp(-E_D/kT))
    """
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    #full ionization solution (from 1st order)
    n_full, p_full = free_carriers_first_order(mat, T, NA, ND)

    #DOS at conduction band edge
    Nc = mat.Nc(T)

    #Safeguards to avoid division by 0 / log of 0 / unrealistic input
    n_safe  = np.maximum(n_full, 1e-30)
    Nc_safe = np.maximum(Nc,      1e-30)

    E_D_J = E_D_eV * q #conversion from eV to J
    # exponent = (E_F - E_D)/kT = ln(n/Nc) - E_D/(kT)
    exponent = np.log(n_safe / Nc_safe) + E_D_J / (kB * T)

    # Ionized frac:
    fD = 1.0 / (1.0 + gD_local * np.exp(exponent))

    return fD, n_full, p_full #return vals






def acceptor_ionization_fraction_with_Fermi(mat, T, NA, ND,
                                            E_A_eV=0.045,
                                            gA_local=0.5):
    """
    Acceptor ionization fraction f_A = N_A^- / N_A including
    the Fermi-level shift due to doping.

    Similar to donors, using:
        p ≈ N_v exp(-(E_F - E_v)/kT) => E_F - E_A = kT ln(p/N_v) - E_A

    and
        N_A^- = N_A / (1 + g_A exp((E_A - E_F)/kT))
               = N_A / (1 + g_A * (p/N_v) * exp(-E_A/kT))
    """
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    # full ionization solution (reuse from earlier)
    n_full, p_full = free_carriers_first_order(mat, T, NA, ND)

    Nv = mat.Nv(T) #uses object corresponding to material Nv as function of T

    #safeguards
    p_safe  = np.maximum(p_full, 1e-30)
    Nv_safe = np.maximum(Nv,     1e-30)

    E_A_J = E_A_eV * q
    # exponent = (E_A - E_F)/kT = ln(p/Nv) - E_A/(kT)
    exponent = np.log(p_safe / Nv_safe) + E_A_J / (kB * T)

    fA = 1.0 / (1.0 + gA_local * np.exp(exponent))

    return fA, n_full, p_full #return values






def free_carriers_incomplete_ionization(mat, T, NA, ND,
                                        E_D_eV=0.045,
                                        E_A_eV=0.045):
    """Now, ionization is incomplete,"""
    T  = np.asarray(T,  dtype=float) #array of Temp
    NA = np.asarray(NA, dtype=float) #array of NA
    ND = np.asarray(ND, dtype=float) #array of ND
    #case-by-case for each material using if-else
    if mat is Si:
        E_D_use, E_A_use = E_D_Si,  E_A_Si
    elif mat is SiC:
        E_D_use, E_A_use = E_D_SiC, E_A_SiC
    elif mat is InAs:
        E_D_use, E_A_use = E_D_InAs, E_A_InAs
    else:
        # fallback to default values specified in function definition
        E_D_use, E_A_use = E_D_eV, E_A_eV
    fD, n_full_D, p_full_D = donor_ionization_fraction_with_Fermi(
        mat, T, NA, ND, E_D_eV=E_D_use, gD_local=gD
    )
    fA, n_full_A, p_full_A = acceptor_ionization_fraction_with_Fermi(
        mat, T, NA, ND, E_A_eV=E_A_use, gA_local=gA
    )

    ND_eff = ND * fD
    NA_eff = NA * fA

    # Another 1st-order solving with effective doping levels
    n_freeze, p_freeze = free_carriers_first_order(mat, T, NA_eff, ND_eff)

    return n_freeze, p_freeze, ND_eff, NA_eff



#plotting

def plot_freezeout_ratio_map(mat,
                             ND_vals,
                             T_vals,
                             NA_fixed=0.0,
                             title_prefix="",
                             fname="freezeout_map.png"):

    # build ND–T grids, NA fixed
    ND_grid, T_grid = np.meshgrid(ND_vals, T_vals, indexing="ij")
    NA_grid = np.full_like(ND_grid, NA_fixed)

    #fully ionized (original 1st order)
    n_full, _ = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    #incomplete ionization (Extra)
    n_freeze, _, ND_eff, NA_eff = free_carriers_incomplete_ionization(
        mat, T_grid, NA_grid, ND_grid
    )

    # ratio -> highlights freeze-out (values < 1), complete if = 1
    #also safeguard against unrealistically low ratio
    ratio = n_freeze / np.maximum(n_full, 1e-30)

    plt.figure(figsize=(9, 6))
    # color scale from 0 to 1
    levels = np.linspace(0.0, 1.0, 31)
    cs = plt.contourf(ND_grid, T_grid, ratio, levels=levels, cmap="viridis")
    #plot logic
    plt.xscale("log")
    plt.xlabel(r"$N_D$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Freeze-Out Ratio  $n_{{freeze}}/n_{{full}}$")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$n_{\mathrm{freeze}} / n_{\mathrm{full}}$")

    # contour line at ratio = 0.5 as a "freeze-out boundary"
    plt.contour(ND_grid, T_grid, ratio, levels=[0.5],
                colors="red", linewidths=1.0)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


#Run for all three conditions: A (Si), B (4H-SiC), C (InAs)

if __name__ == "__main__":
    # Common ND axis
    #using extremely low temperatures, maxing out at room temperature 300K, interesting to analyze
    #NOTE: also did same thing with varying temperature windows corresponding to each material's bandgaps
    T_axis = np.linspace(0, 300, 300) 
    ND_vals = np.logspace(13, 20, 60)

    #Silicon
    T_vals_Si = T_axis
    plot_freezeout_ratio_map(Si,
                             ND_vals=ND_vals,
                             T_vals=T_vals_Si,
                             NA_fixed=0.0,
                             title_prefix="Silicon:",
                             fname="fig_Si_freezeout_ratio_LowTonly.png")

    #4H-SiC
    T_vals_SiC = T_axis
    plot_freezeout_ratio_map(SiC,
                             ND_vals=ND_vals,
                             T_vals=T_vals_SiC,
                             NA_fixed=0.0,
                             title_prefix="4H-SiC:",
                             fname="fig_SiC_freezeout_ratio_LowTonly.png")

    #InAs
    T_vals_InAs = T_axis
    plot_freezeout_ratio_map(InAs,
                             ND_vals=ND_vals,
                             T_vals=T_vals_InAs,
                             NA_fixed=0.0,
                             title_prefix="InAs:",
                             fname="fig_InAs_freezeout_ratio_LowTonly.png")

   
