import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = [
    {'parameter': 'phi_backfill', 'min': 25, 'mode':30, 'max':35},
    {'parameter': 'phi_cs', 'min':19, 'mode':20, 'max':21},
    {'parameter': 'depth_to_cs', 'min':0.5, 'mode':5, 'max':7},
    {'parameter': 'weight_backfill', 'min':105, 'mode':120, 'max':125}       
]

df = pd.DataFrame(data)
print(df)


# monolith
top_of_wall = 606.5 # feet
bottom_of_wall = 555.5 # feet
H = top_of_wall - bottom_of_wall # feet
top_width = 6 # feet
base_width = 26 # feet
heel_height = 12 # feet
mono_area = (base_width * heel_height) + (top_width * (H - heel_height)) + (0.5 * (H - heel_height) * (base_width - top_width))
# area in SF
g_w = 62.4 # PCF unit weight of water
g_c = 150 # PCF unit weight of concrete
# SF * PCF = lbf/ft
# divide by 1000 to get kips/ft
mono_weight = mono_area * g_c / 1000
print("N = {} kips/ft".format(mono_weight))


# driving forces
# active soil


# Function to calculate Factor of Safety (FoS) against sliding
def calculate_fos(sliding_force, resisting_force):
    return resisting_force / sliding_force

# Monte Carlo Simulation parameters
num_simulations = 10000  # Number of Monte Carlo simulations

# Define triangular distribution parameters: (min, most likely, max)
soil_phi_params = (25, 30, 35)  # degrees
soil_unit_weight_params = (105, 120, 125)  # pcf
foundation_phi_params = (19, 20, 35)  # degrees

# Define constant parameters
monolith_weight = mono_weight  # kip/ft
cohesion = 0  # kPa (if cohesion is negligible)

# Arrays to store results
fos_values = []

# Perform Monte Carlo Simulation
for _ in range(num_simulations):
    # Sample input parameters using triangular distribution
    soil_phi = np.random.triangular(*soil_phi_params)
    soil_unit_weight = np.random.triangular(*soil_unit_weight_params)
    foundation_phi = np.random.triangular(*foundation_phi_params)
    
    
    # Convert angles from degrees to radians for calculations
    soil_phi_rad = np.radians(soil_phi)
    foundation_phi_rad = np.radians(foundation_phi)
    
    # Calculate resisting force (simplified)
    resisting_force = monolith_weight * np.tan(foundation_phi_rad)
    
    # Calculate sliding force
    radians45 = np.radians(45)
    Ka = (np.tan(radians45 - (soil_phi_rad / 2) )) ** 2
    sliding_force = 1/2 * Ka * soil_unit_weight * ((H - heel_height) ** 2) / 1000
    
    # Calculate Factor of Safety (FoS)
    fos = calculate_fos(sliding_force, resisting_force)
    
    # Store result
    fos_values.append(fos)

# Analyze results
fos_values = np.array(fos_values)
mean_fos = np.mean(fos_values)
std_fos = np.std(fos_values)

# Print statistical summary
print(f"Monte Carlo Simulation Results:")
print(f"Mean Factor of Safety: {mean_fos:.2f}")
print(f"Standard Deviation of FoS: {std_fos:.2f}")
print(f"Minimum FoS: {np.min(fos_values):.2f}")
print(f"Maximum FoS: {np.max(fos_values):.2f}")

print("For phi = {}, Ka = {}".format(soil_phi, Ka))

# Plot histogram of FoS values
plt.hist(fos_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Factor of Safety Against Sliding")
plt.xlabel("Factor of Safety")
plt.ylabel("Frequency")
plt.axvline(mean_fos, color='red', linestyle='dashed', linewidth=1, label=f"Mean FoS = {mean_fos:.2f}")
plt.legend()
plt.show()
