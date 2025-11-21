import numpy as np  #import matplotlib.pyplot as plt
from  First_Order import free_carriers_first_order, kB #import functions from first_order
from First_Order import Si, SiC, InAs # import material objects from the first order 
# Degeneracy threshold and conditions
def second_order_free_carriers(material, T, NA, ND, degeneracy_threshold=3.0):
    """
    second order function using first order free carrier solution 
    
    This does not plot, it:
    1) calls the intial first order function to get n and p
    2) computes Ef - Ei for electrons using kT*ln(n/ni)
    3) builds a grid that maps for which points on the map using the temperature/doping combinations are degenerate and non-degenerate

    Parameters:
    material - object (e.g,, Si, SiC, InAs) that must provide ni(T) method
        T: array-like/float 
            Temperature(s) I would like to check for degeneracy, measured in Kelvin
        NA: array-like/float
            How much P-type doping there is (acceptors) 
        ND: array-like/float
            How much N-type doping there is (donors)
        degeneracy_threshhold: float
            Sets the cuttoff for deciding when the material becomes degenerate.
            Usually use 3*KT based off class notes
    """
    
    T = np.asarray(T, dtype=float)  # converts the temeperature input into an array taken as a decimal value
    NA = np.asarray(NA, dtype=float) # converts the acceptor input into an array taken as a decimal value
    ND = np.asarray(ND, dtype=float) # converts the donor input into an array taken as a decimal value
    
    n,p = free_carriers_first_order(material, T, NA, ND) # calls the first order function to 

    ni = material.ni(T) # gets the intrinsic carrier concentration for the material at temperature T 

    n_constraint = np.where(n <= 0, 1e-30,n) #avoids log of zero by replacing any n values that are less than or equal to zero with a small number
    ni_constraint = np.where(ni <= 0, 1e-30, ni) #avoids log of zero by replacing any ni values that are less than or equal to zero with a small number
    
    Ef_minus_Ei = kB * T * np.log(n_constraint / ni_constraint) # computes the difference between the free electrons and free holes using the formula kT*ln(n/ni) 

    degenerate_map = Ef_minus_Ei > degeneracy_threshold * kB * T # Identfies the temperature-doping combination where the material becomes degenerate
    
    return n, p, Ef_minus_Ei, degenerate_map # returns the free electron concentration, free hole conentration, Ef - Ei values, and the degenerate map



#calling the function to test it out 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print('running second order main ...') # verifying the plot function is running 

    # Choos Teamperature (T) and doping axes once here for all plots
    T_axis = np.linspace(300, 3200, 300) # K
    ND_axis = np.logspace(15, 20, 60) # cm^-3
    T_grid, ND_grid = np.meshgrid(T_axis, ND_axis, indexing="ij")#creates a grid of temperature and donor concentration values
    NA_grid = np.zeros_like(T_grid)# creates a grid of acceptor concentration values set to zero (n-type doping only)

    materials = [Si, SiC, InAs] # List of material obkects that we imported from First_Order.py
    names = ["Si", "SiC", "InAs"] # List of strings for the material names
    #Loop over materials that are lists to generate plots for each material
    for mat, name in zip(materials, names):
        n,p, Ef_minus_Ei , deg_map = second_order_free_carriers(mat, T_grid, NA_grid, ND_grid)

        T_melt_dict = {"Si": 1687, "SiC": 3100, "InAs": 1215}# Melting temperatures for each material in Kelvin from online resources
        T_melt = T_melt_dict[name]# Assigning melting temperature based on material name

        melt_mask = T_grid >= T_melt # Creating a binary map (0/1) where the temaperature exceeds the melting point 
        #plot or print results
        plt.figure()
        cs = plt.contourf(ND_axis, T_axis, deg_map, levels=[0 , 0.5, 1] )
        plt.title(f"Second-Order Ef - Ei for {name}") # Defining the title of the plot
        plt.colorbar(cs, label="Non-degenerate (0) / Degenerate (1)") # Adding a color bar for dgenerate and non-degenerate 
        plt.xscale("log") # Setting the x-axis not to be 1,2,3,4,5,etc. but rather logarithmic scale like 10,100,1000,10000,etc.
        plt.xlabel("Donor Concentration ND (cm$^{-3}$)") # lables the x-axis in terms of the Number of Doners(Note the $ is latex so that the labels come out looking cleaner)
        plt.ylabel("Temperature (K)")#Lables the y-axis in terms of temperature in Kelvin

        plt.contour(ND_grid, T_grid, deg_map, levels = [0.5], colors="red") # Adding a contour line to show the boundary betweeen degenerate and non-degenerate regions
        plt.contour(ND_grid, T_grid, melt_mask, levels = [0.5], colors = "black", linestyles="dashed") # Adding a contour line thats a black dashed line to show the melting temperature boundary
        
    plt.show() # Finally show all the plots once they have been generated