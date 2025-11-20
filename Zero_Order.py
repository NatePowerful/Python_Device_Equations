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

if __name__ == "__main__":
    import matplotlib.pyplot as plt # importing matplotlib for plotting
    T_axis = np.linspace(250,1200,100) #Line plots of the temperature range from 250K to 1200K in 110 steps 

    materials = [Si, SiC, InAs]# lists of the material objects we created above 
    names = ["Si","SiC","InAs"] # list of strings for the material names

    plt.figure() # Starts the plotting figure 
    for mat, name in zip(materials,names):# Looping over the material objects and their names associated with them
        ni_T = mat.ni(T_axis) # calling the ni method from each material object to get intrinsic carrier concentration as a function of temperature
        plt.semilogy(T_axis,ni_T, label = name) #plotting the intrinsic carrier concentration vs temperature on a 2D semilog graph 
        plt.xlabel("Temperature (K)") #labels the x-axis 
        plt.ylabel("Intrinsic Carrier Concentration ni (cm$^{-3}$)") #labels the y-axis
        plt.title(f"Zeroth-Order Intrinsic Carrier Concentration for {name}")#title of the actual plot
        plt.grid( True, which = "both", ls = "--") #adding a grid to the plot for better vsual clarity)) 
        plt.show() # finally show the plot once all the materials have been plotted