import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# define number of simulations
num = 10000


# Function to calculate Factor of Safety (FoS) against sliding
def calculate_FS(N, phi, T):
    FS = (N * np.tan(np.radians(phi))) / T
    return FS


# CONSTANT parameters
el_top = 692.71         # ft,   elevation of top of cell
phi_c = 35              # degrees, sliding friction for concrete on clean rock
gamma_w = 62.4          # pcf,  unit weight of water
D = 72.08               # ft,   cell diameter
A = 5627                # sf,   area of one cell and one connecting arc
y = 92.36               # ft,   center-center spacing of cells
Df = A / y              # ft,   equivalent cell width


# DETERMINISTIC parameters
el_w = 691.71         # ft,   elevation of water surface
el_tor = 620            # ft,   foundation elevation
el_coal = 615           # ft, elevation of coal seam
phi_a = 34              # degrees, angle of internal friction of alluvium overburden
phi_coal = 21           # degrees, sliding friction for coal seam
phi_cbs = 26            # degrees, cross bed shear sliding friction
gamma_r = 160           # pcf, unit weight of rock
gamma_a = 115            # pcf,  unit weight of alluvium overburden
gamma_c = 145           # pcf,  unit weight of concrete
H_c = el_top - el_tor       # ft,   height of cell
H_w = el_w - el_tor      # ft,   height of water column to foundation elevation
H_t = 5                 # ft,    tremie seal thickness
H_f = H_c - H_t        # ft,   height of alluvium fill within cofferdam cell
H_o = 30                # ft, overburden thickness on driving side
H_r = el_tor - el_coal  # ft, height of rock in structural wedge
H_w_dss = el_w - el_coal  # ft, height of water column for deep seated sliding


### SLIDING ON THE BASE ###
# calculate weight of cell
# first calc weight of cell fill
W_a = (gamma_a * (H_f - 5) * Df + (gamma_a - gamma_w) * 5 * Df) / 1000  # kips/ft
# next calc weight of tremie seal
W_tremie = gamma_c * H_t * Df / 1000  # kips/ft
# then calc total weight of cell
W_t = W_a + W_tremie  # kips/ft
# calculate lateral forces acting on structural wedge
P_w = 0.5 * gamma_w * (H_w ** 2) / 1000  # kips/ft, hydrostatic force
K_a = np.tan(np.radians(45 - (phi_a / 2))) ** 2  # active earth pressure coefficient
P_a = 0.5 * K_a * (gamma_a - gamma_w) * (H_o ** 2) / 1000  # kips/ft, driving soil force
# calculate uplift
u_heel = gamma_w * H_w  # psf, uplift pressure at heel
u_toe = 0               # psf, uplift pressure at toe
U = 0.5 * (u_heel + u_toe) * Df * 0.5 / 1000    # kips/ft, uplift force
# note uplift force allowed to be reduced by 50% due to relief wells
# calculate driving forces
T = P_w + P_a  # kips/ft 
# calculate resisting forces
N = W_t - U     # kips/ft
# calculate FS against sliding on the base
calculate_FS(N, phi_c, T)
FS = (N * np.tan(np.radians(phi_c))) / T
print(f"FS = {FS} for sliding on the base.")


### DEEP SEATED SLIDING ###
W_r = gamma_r * H_r * Df / 1000  # kips/ft, weight of rock in structural wedge
W_t = W_t + W_r  # kips/ft, total weight of structural wedge for DEEP SEATED SLIDING
P_w = 0.5 * gamma_w * (H_w_dss ** 2) / 1000  # kips/ft, hydrostatic force
alpha = 30          # degrees, inclination angle of passive rock wedge
# note that alpha is found by trial and error in the original analysis
# consider making alpha a variable between 25-45 degrees
w_p = H_r / np.tan(np.radians(alpha))  # ft, width of passive rock wedge
W_p = 0.5 * w_p * gamma_r * H_r / 1000   # kips/ft, weight of passive rock wedge
L_p = H_r / np.sin(np.radians(alpha))  # ft, L of failure plane beneath passive wedge
u_heel = gamma_w * H_w_dss  # psf, uplift pressure at sliding plane below heel
u_toe = gamma_w * H_r       # psf, uplift pressure at sliding plane below toe
U_t = (u_heel + u_toe) * 0.5 * Df * 0.5 / 1000 # kips/ft, uplift on sructural wedge
U_p = u_toe * 0.5 * L_p * 0.5 / 1000  # kips/ft, uplift on passive wedge
T = P_w + P_a - (W_p * np.tan(np.radians(alpha)))   # kips/ft driving forces
R = (W_t - U_t) * np.tan(np.radians(phi_coal)) + (W_p - (U_p * np.cos(np.radians(alpha)) * np.tan(np.radians(phi_cbs))))     # kips/ft resisting forces
FS = R / T
print(f"FS = {FS} for deep seated sliding.") 


# Arrays to store results of Monte Carlo simulations
FS_values = []
el_tor_values = []
el_coal_values = []
phi_a_values = []
phi_coal_values = []
gamma_a_values = []
gamma_r_values = []
H_c_values = []
H_o_values = []
FS_with_piles_values = []
R_without_piles_values = []
R_with_piles_values = []
T_values = []

# Perform Monte Carlo Simulation
for _ in range(num):
    # define variable parameters according to their distribution
    el_tor = np.random.normal(620, 0.5)
    depth_to_coal = np.random.triangular(0, 5, 10)
    el_coal = el_tor - depth_to_coal
    phi_a = np.random.triangular(28, 34, 36)
    phi_coal = np.random.triangular(20, 21, 22)
    phi_cbs = np.random.triangular(26, 27, 28)
    gamma_a = np.random.normal(122, 6.4)
    gamma_c = np.random.normal(145, 1.4)
    gamma_r = np.random.normal(165, 5)
    H_c = el_top - el_tor
    H_w = el_w - el_coal
    H_f = H_c - H_t
    H_o = np.random.normal(27, 3)
    H_r = depth_to_coal
    phi_sliding = phi_coal
    if depth_to_coal == 0:
        phi_sliding = phi_c


    # calculate FS for this iteration
    # if depth to coal = zero, calculate FS against sliding on the base
    # otherwise it is deep seated sliding
    W_a = (gamma_a * (H_f - 5) * Df + (gamma_a - gamma_w) * 5 * Df) / 1000  # kips/ft
    W_r = gamma_r * H_r * Df / 1000  # kips/ft, weight of rock in structural wedge
    W_t = W_a + W_tremie + W_r  # kips/ft
    P_w = 0.5 * gamma_w * (H_w_dss ** 2) / 1000  # kips/ft, hydrostatic force
    alpha = 30          # degrees, inclination angle of passive rock wedge
    if depth_to_coal == 0:
        alpha = 0
    w_p = H_r / np.tan(np.radians(alpha))  # ft, width of passive rock wedge
    W_p = 0.5 * w_p * gamma_r * H_r / 1000   # kips/ft, weight of passive rock wedge
    L_p = H_r / np.sin(np.radians(alpha))  # ft, L of failure plane beneath passive wedge
    u_heel = gamma_w * H_w_dss  # psf, uplift pressure at sliding plane below heel
    u_toe = gamma_w * H_r       # psf, uplift pressure at sliding plane below toe
    U_t = (u_heel + u_toe) * 0.5 * Df * 0.5 / 1000 # kips/ft, uplift on sructural wedge
    U_p = u_toe * 0.5 * L_p * 0.5 / 1000  # kips/ft, uplift on passive wedge
    T = P_w + P_a - (W_p * np.tan(np.radians(alpha)))   # kips/ft driving forces
    R = (W_t - U_t) * np.tan(np.radians(phi_sliding)) + (W_p - (U_p * np.cos(np.radians(alpha)) * np.tan(np.radians(phi_cbs)))) 
    FS = R / T

    # add the resistance from foundation piles and recalculate FS
    V_a = 510  # kips, shear capacity of one pile
    N_piles = 19  # number of piles in the cell
    R_piles = (V_a * N_piles) / y  # kips/ft, resistance from foundation piles
    new_R = R + R_piles
    FS_with_piles = (R + R_piles) / T

    # Store result
    FS_values.append(FS)
    el_tor_values.append(el_tor)
    el_coal_values.append(el_coal)
    phi_a_values.append(phi_a)
    phi_coal_values.append(phi_coal)
    gamma_a_values.append(gamma_a)
    gamma_r_values.append(gamma_r)
    H_c_values.append(H_c)
    H_o_values.append(H_o)
    FS_with_piles_values.append(FS_with_piles)
    R_without_piles_values.append(R)
    R_with_piles_values.append(new_R)
    T_values.append(T)

# Analyze results
FS_values = np.array(FS_values)
mean_FS = np.mean(FS_values)
std_FS = np.std(FS_values)
FS_with_piles_values = np.array(FS_with_piles_values)
mean_FS_with_piles = np.mean(FS_with_piles_values)
std_FS_with_piles = np.std(FS_with_piles_values)

# Calculate probability of failure (FoS < 1)
prob_failure = np.sum(FS_values < 1) / len(FS_values)
print(f"Probability of Failure (FoS < 1): {prob_failure:.2%}")

# For FoS with piles included
prob_failure_with_piles = np.sum(FS_with_piles_values < 1) / len(FS_with_piles_values)
print(f"Probability of Failure with Piles (FoS < 1): {prob_failure_with_piles:.10%}")

# Print statistical summary
print(f"Monte Carlo Simulation Results:")
print(f"Mean Factor of Safety: {mean_FS:.2f}")
print(f"Standard Deviation of FoS: {std_FS:.2f}")
print(f"Minimum FoS: {np.min(FS_values):.2f}")
print(f"Maximum FoS: {np.max(FS_values):.2f}")

# now repeat for with piles included
print(f"Monte Carlo Simulation Results with piles included:")
print(f"Mean Factor of Safety: {mean_FS_with_piles:.2f}")
print(f"Standard Deviation of FoS: {std_FS_with_piles:.2f}")
print(f"Minimum FoS: {np.min(FS_with_piles_values):.2f}")
print(f"Maximum FoS: {np.max(FS_with_piles_values):.2f}")

# Plot histogram of FoS values
plt.hist(FS_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Factor of Safety Against Sliding")
plt.xlabel("Factor of Safety")
plt.ylabel("Frequency")
plt.axvline(mean_FS, color='red', linestyle='dashed', linewidth=1, label=f"Mean FS = {mean_FS:.2f}")
plt.legend()
plt.show()

# Plot histogram for R and T
plt.hist(R_without_piles_values, bins=30, color='blue', alpha=0.7, label='Resisting Forces (R)')
plt.hist(T_values, bins=30, color='orange', alpha=0.7, label='Driving Forces (T)')
plt.title("Monte Carlo Simulation: Resisting vs Driving Forces")
plt.xlabel("Force (kips/ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(R_without_piles_values), color='blue', linestyle='dashed', linewidth=1, label=f"Mean R = {np.mean(R_without_piles_values):.2f} kips/ft")
plt.axvline(np.mean(T_values), color='orange', linestyle='dashed', linewidth=1, label=f"Mean T = {np.mean(T_values):.2f} kips/ft")
plt.legend()
plt.show()

# Plot histogram of FoS values with piles
plt.hist(FS_with_piles_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Factor of Safety Against Sliding with piles included")
plt.xlabel("Factor of Safety")
plt.ylabel("Frequency")
plt.axvline(mean_FS_with_piles, color='red', linestyle='dashed', linewidth=1, label=f"Mean FS w/ Piles= {mean_FS_with_piles:.2f}")
plt.legend()
plt.show()

# Plot histogram for new_R and T
plt.hist(R_with_piles_values, bins=30, color='blue', alpha=0.7, label='New Resisting Forces (R)')
plt.hist(T_values, bins=30, color='orange', alpha=0.7, label='Driving Forces (T)')
plt.title("Monte Carlo Simulation: New Resisting vs Driving Forces")
plt.xlabel("Force (kips/ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(R_with_piles_values), color='blue', linestyle='dashed', linewidth=1, label=f"Mean R = {np.mean(R_with_piles_values):.2f} kips/ft")
plt.axvline(np.mean(T_values), color='orange', linestyle='dashed', linewidth=1, label=f"Mean T = {np.mean(T_values):.2f} kips/ft")
plt.legend()
plt.show()

# plot histogram of top of rock elevation
plt.hist(el_tor_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Top of Rock Elevation")
plt.xlabel("Elevation (ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(el_tor_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Elevation = {np.mean(el_tor_values):.2f} ft")
plt.legend()
plt.show()

# plot histogram of coal elevation
plt.hist(el_coal_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Coal Elevation")
plt.xlabel("Elevation (ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(el_coal_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Elevation = {np.mean(el_coal_values):.2f} ft")
plt.legend()
plt.show()

# plot historgram of alluvium phi angle
plt.hist(phi_a_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Alluvium Phi Angle")
plt.xlabel("Phi Angle (degrees)")
plt.ylabel("Frequency")
plt.axvline(np.mean(phi_a_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Phi = {np.mean(phi_a_values):.2f} degrees")
plt.legend()
plt.show()

# plot histogram of coal phi angle
plt.hist(phi_coal_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Coal Phi Angle")
plt.xlabel("Phi Angle (degrees)")
plt.ylabel("Frequency")
plt.axvline(np.mean(phi_coal_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Phi = {np.mean(phi_coal_values):.2f} degrees")
plt.legend()
plt.show()

# plot histogram of alluvium weight
plt.hist(gamma_a_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Alluvium Unit Weight")
plt.xlabel("Unit Weight (pcf)")
plt.ylabel("Frequency")
plt.axvline(np.mean(gamma_a_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Unit Weight = {np.mean(gamma_a_values):.2f} pcf")
plt.legend()
plt.show()

# plot histogram of rock weight
plt.hist(gamma_r_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Rock Unit Weight")
plt.xlabel("Unit Weight (pcf)")
plt.ylabel("Frequency")
plt.axvline(np.mean(gamma_r_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Unit Weight = {np.mean(gamma_r_values):.2f} pcf")
plt.legend()
plt.show()

# plot histogram of cell height
plt.hist(H_c_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Cell Height")
plt.xlabel("Height (ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(H_c_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Height = {np.mean(H_c_values):.2f} ft")
plt.legend()
plt.show()

# plot histogram of overburden height
plt.hist(H_o_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Overburden Height")
plt.xlabel("Height (ft)")
plt.ylabel("Frequency")
plt.axvline(np.mean(H_o_values), color='red', linestyle='dashed', linewidth=1, label=f"Mean Height = {np.mean(H_o_values):.2f} ft")
plt.legend()
plt.show()

