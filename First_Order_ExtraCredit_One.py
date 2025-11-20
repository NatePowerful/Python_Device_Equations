import numpy as np
import matplotlib.pyplot as plt

# import what we need from your existing First_Order.py
from First_Order import (
    free_carriers_first_order,
    kB, q, Tref,
    Si, SiC, InAs
)

# ---------- Incomplete ionization (extra credit #1) ----------
# Shallow-donor / shallow-acceptor energies for each material (in eV)
# Adjust these if your instructor gives more specific values.
E_D_Si   = 0.045   # P in Si ~ 45 meV
E_D_SiC  = 0.200   # rough shallow donor in 4H-SiC (example)
E_D_InAs = 0.006   # very shallow donor in InAs (example)


E_A_Si   = 0.045   # shallow B in Si (example)
E_A_SiC  = 0.200   # rough acceptor in 4H-SiC (example)
E_A_InAs = 0.03   # example shallow acceptor in InAs

# degeneracy factors (typical values)
gD = 0.5
gA = 0.5

# ---------- Incomplete ionization (extra credit #1) ----------


def donor_ionization_fraction_with_Fermi(mat, T, NA, ND,
                                         E_D_eV=0.045,
                                         gD_local=0.5):
    """
    Donor ionization fraction f_D = N_D^+ / N_D including
    the Fermi-level shift due to doping.

    Uses the 1st-order solution to estimate n, then:
        n ≈ N_c exp(-(E_c - E_F)/kT)
        => E_F - E_D = kT ln(n/N_c) - E_D

    and
        N_D^+ = N_D / (1 + g_D exp((E_F - E_D)/kT))
              = N_D / (1 + g_D * (n/N_c) * exp(-E_D/kT))
    """
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    # 1) full-ionization solution (your 1st-order model)
    n_full, p_full = free_carriers_first_order(mat, T, NA, ND)

    # 2) density of states at the conduction band edge
    Nc = mat.Nc(T)

    # Safeguards to avoid division by 0 / log of 0
    n_safe  = np.maximum(n_full, 1e-30)
    Nc_safe = np.maximum(Nc,      1e-30)

    E_D_J = E_D_eV * q
    # exponent = (E_F - E_D)/kT = ln(n/Nc) - E_D/(kT)
    exponent = np.log(n_safe / Nc_safe) + E_D_J / (kB * T)

    # Ionized fraction:
    fD = 1.0 / (1.0 + gD_local * np.exp(exponent))

    return fD, n_full, p_full






def acceptor_ionization_fraction_with_Fermi(mat, T, NA, ND,
                                            E_A_eV=0.045,
                                            gA_local=0.5):
    """
    Acceptor ionization fraction f_A = N_A^- / N_A including
    the Fermi-level shift due to doping.

    Analogous to donors, using:
        p ≈ N_v exp(-(E_F - E_v)/kT)
        => E_F - E_A = kT ln(p/N_v) - E_A

    and
        N_A^- = N_A / (1 + g_A exp((E_A - E_F)/kT))
               = N_A / (1 + g_A * (p/N_v) * exp(-E_A/kT))
    """
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    # 1) full-ionization solution (reuse from donor call or recalc)
    n_full, p_full = free_carriers_first_order(mat, T, NA, ND)

    Nv = mat.Nv(T)

    p_safe  = np.maximum(p_full, 1e-30)
    Nv_safe = np.maximum(Nv,     1e-30)

    E_A_J = E_A_eV * q
    # exponent = (E_A - E_F)/kT = ln(p/Nv) - E_A/(kT)
    exponent = np.log(p_safe / Nv_safe) + E_A_J / (kB * T)

    fA = 1.0 / (1.0 + gA_local * np.exp(exponent))

    return fA, n_full, p_full






def free_carriers_incomplete_ionization(mat, T, NA, ND,
                                        E_D_eV=0.045,
                                        E_A_eV=0.045):
    T  = np.asarray(T,  dtype=float)
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)

    if mat is Si:
        E_D_use, E_A_use = E_D_Si,  E_A_Si
    elif mat is SiC:
        E_D_use, E_A_use = E_D_SiC, E_A_SiC
    elif mat is InAs:
        E_D_use, E_A_use = E_D_InAs, E_A_InAs
    else:
        # fallback to user-supplied or default values
        E_D_use, E_A_use = E_D_eV, E_A_eV
    fD, n_full_D, p_full_D = donor_ionization_fraction_with_Fermi(
        mat, T, NA, ND, E_D_eV=E_D_use, gD_local=gD
    )
    fA, n_full_A, p_full_A = acceptor_ionization_fraction_with_Fermi(
        mat, T, NA, ND, E_A_eV=E_A_use, gA_local=gA
    )

    ND_eff = ND * fD
    NA_eff = NA * fA

    # One more 1st-order solve with effective dopings
    n_freeze, p_freeze = free_carriers_first_order(mat, T, NA_eff, ND_eff)

    return n_freeze, p_freeze, ND_eff, NA_eff


# ---------- Contour plot: freeze-out ratio map ----------

def plot_freezeout_ratio_map(mat,
                             ND_vals,
                             T_vals,
                             NA_fixed=0.0,
                             title_prefix="",
                             fname="freezeout_map.png"):

    # build ND–T grids
    ND_grid, T_grid = np.meshgrid(ND_vals, T_vals, indexing="ij")
    NA_grid = np.full_like(ND_grid, NA_fixed)

    # 1) fully ionized (your original 1st order)
    n_full, _ = free_carriers_first_order(mat, T_grid, NA_grid, ND_grid)

    # 2) incomplete ionization (extra credit)
    n_freeze, _, ND_eff, NA_eff = free_carriers_incomplete_ionization(
        mat, T_grid, NA_grid, ND_grid
    )

    # 3) ratio -> highlights freeze-out (values < 1)
    ratio = n_freeze / np.maximum(n_full, 1e-30)

    plt.figure(figsize=(9, 6))
    # color scale from 0 to 1
    levels = np.linspace(0.0, 1.0, 31)
    cs = plt.contourf(ND_grid, T_grid, ratio, levels=levels, cmap="viridis")

    plt.xscale("log")
    plt.xlabel(r"$N_D$ (cm$^{-3}$)")
    plt.ylabel("Temperature (K)")
    plt.title(f"{title_prefix} Freeze-Out Ratio  $n_{{freeze}}/n_{{full}}$")

    cbar = plt.colorbar(cs)
    cbar.set_label(r"$n_{\mathrm{freeze}} / n_{\mathrm{full}}$")

    # Optional: draw a contour line at ratio = 0.5 as a "freeze-out boundary"
    plt.contour(ND_grid, T_grid, ratio, levels=[0.5],
                colors="red", linewidths=1.0)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# ---------- Run for all three conditions: A (Si), B (4H-SiC), C (InAs) ----------

if __name__ == "__main__":
    # Common ND axis
    T_axis = np.linspace(100, 3000, 300) 
    ND_vals = np.logspace(13, 20, 60)

    # Condition A: Silicon (similar window to your 1st-order Si plots, but extend down to low T)
    T_vals_Si = T_axis
    plot_freezeout_ratio_map(Si,
                             ND_vals=ND_vals,
                             T_vals=T_vals_Si,
                             NA_fixed=0.0,
                             title_prefix="Silicon:",
                             fname="fig_Si_freezeout_ratio.png")

    # Condition B: 4H-SiC (higher max temp, but we still care about low-T freeze-out)
    T_vals_SiC = T_axis
    plot_freezeout_ratio_map(SiC,
                             ND_vals=ND_vals,
                             T_vals=T_vals_SiC,
                             NA_fixed=0.0,
                             title_prefix="4H-SiC:",
                             fname="fig_SiC_freezeout_ratio.png")

    # Condition C: InAs (lower max temp)
    T_vals_InAs = T_axis
    plot_freezeout_ratio_map(InAs,
                             ND_vals=ND_vals,
                             T_vals=T_vals_InAs,
                             NA_fixed=0.0,
                             title_prefix="InAs:",
                             fname="fig_InAs_freezeout_ratio.png")

   
