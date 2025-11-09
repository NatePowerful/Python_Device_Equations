import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass


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
# Input validation test for user inputs
# -------------------------
def validate_inputs(T, NA, ND):
    if np.any(T < 0):
        raise ValueError("Temperature must be ≥ 0 K.")
    if np.any(NA < 0) or np.any(ND < 0):
        raise ValueError("Dopant concentrations must be ≥ 0 cm^-3.")
    if np.any(NA > 1e21) or np.any(ND > 1e21):
        print("Error: doping levels above ~1e21 cm^-3 are nonphysical for most semiconductors.")

# -------------------------
# DOS helpers to find Nc and Nv
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
