import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pulp
from math import radians, sin, cos, sqrt, atan2
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, kstest
import statsmodels.api as sm
from io import BytesIO
import base64

# =============================================================================
# PART 1: Create a Random CSV File for Medicine Supply Chain
# =============================================================================
np.random.seed(42)  # For reproducibility

rows = []
# Create 4 warehouse nodes.
for i in range(4):
    node_id = f"W{i+1}"
    node_type = "warehouse"
    # Generate latitude and longitude (roughly within US bounds)
    latitude = np.random.uniform(30, 50)
    longitude = np.random.uniform(-120, -70)
    supply_value = np.random.randint(800, 1200)  # Supply units
    demand_value = 0
    region = ""  # warehouses have no region tag
    rows.append([node_id, node_type, latitude, longitude, supply_value, demand_value, region])

# Create 6 demand nodes.
regions = ["NE_Corridor", "Pacific_NW", "Deep_South"]
for i in range(6):
    node_id = f"D{i+1}"
    node_type = "demand"
    latitude = np.random.uniform(25, 50)
    longitude = np.random.uniform(-125, -70)
    supply_value = 0
    demand_value = np.random.randint(200, 800)  # Demand units
    region = np.random.choice(regions)
    rows.append([node_id, node_type, latitude, longitude, supply_value, demand_value, region])

# Create a DataFrame and save as CSV.
df_csv = pd.DataFrame(rows, columns=["node_id", "node_type", "latitude", "longitude", "supply", "demand", "region"])
csv_filename = "medicine_supply_chain.csv"
df_csv.to_csv(csv_filename, index=False)

print("CSV file head:")
print(pd.read_csv(csv_filename).head())

# =============================================================================
# PART 2: Supply Chain Optimization & Mapping Using the CSV Data
# =============================================================================
data = pd.read_csv(csv_filename)
warehouses = data[data['node_type'] == 'warehouse'].copy()
demand_nodes = data[data['node_type'] == 'demand'].copy()

# Create dictionaries for supply and demand keyed by node_id.
supply_dict = warehouses.set_index('node_id')['supply'].to_dict()
demand_dict = demand_nodes.set_index('node_id')['demand'].to_dict()

# --- Define a cost function using the Haversine distance ---
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Build cost dictionary.
costs = {}
for idx, row_w in warehouses.iterrows():
    for jdx, row_d in demand_nodes.iterrows():
        c = haversine(row_w['longitude'], row_w['latitude'],
                        row_d['longitude'], row_d['latitude'])
        costs[(row_w['node_id'], row_d['node_id'])] = c

# --- Set up the optimization problem using PuLP ---
prob = pulp.LpProblem("Medicine_Supply_Chain_Optimization", pulp.LpMinimize)

# Decision variables: shipment quantities from each warehouse to each demand node.
routes = pulp.LpVariable.dicts("Shipments",
                                 (supply_dict.keys(), demand_dict.keys()),
                                 lowBound=0, cat='Continuous')

# Objective: minimize total transportation cost.
prob += pulp.lpSum([routes[i][j] * costs[(i, j)]
                    for i in supply_dict.keys()
                    for j in demand_dict.keys()]), "Total_Transportation_Cost"

# Supply constraints.
for i in supply_dict.keys():
    prob += pulp.lpSum([routes[i][j] for j in demand_dict.keys()]) <= supply_dict[i], f"Supply_{i}"

# Demand constraints.
for j in demand_dict.keys():
    prob += pulp.lpSum([routes[i][j] for i in supply_dict.keys()]) >= demand_dict[j], f"Demand_{j}"

prob.solve()

# Collect shipment results.
shipment_results = []
for i in supply_dict.keys():
    for j in demand_dict.keys():
        qty = routes[i][j].varValue
        if qty and qty > 1e-6:
            shipment_results.append((i, j, qty))

print("\nSupply Chain Optimization Results")
print("Status:", pulp.LpStatus[prob.status])
print("Total Minimum Transportation Cost: {:.2f}".format(pulp.value(prob.objective)))
print("Shipment Plan:")
for (w, d, q) in shipment_results:
    print(f"  Ship from {w} to {d}: {q:.2f} units")

# --- Create an interactive map with Folium ---
mean_lat = data['latitude'].mean()
mean_lon = data['longitude'].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=5)

# Add warehouse markers.
for idx, row in warehouses.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Warehouse {row['node_id']}<br>Supply: {row['supply']} units",
        icon=folium.Icon(color='blue', icon='building')
    ).add_to(m)

# Add demand node markers.
for idx, row in demand_nodes.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"Facility {row['node_id']}<br>Demand: {row['demand']} units<br>"
                f"Region: {row['region']}"),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

# Draw shipment routes.
for i, j, qty in shipment_results:
    supply_coords = warehouses[warehouses['node_id'] == i][['latitude', 'longitude']].values[0]
    demand_coords = demand_nodes[demand_nodes['node_id'] == j][['latitude', 'longitude']].values[0]
    folium.PolyLine(
        locations=[(supply_coords[0], supply_coords[1]),
                   (demand_coords[0], demand_coords[1])],
        color="green", weight=2, opacity=0.8,
        popup=f"Ship {qty:.2f} units"
    ).add_to(m)

map_filename = "supply_chain_map.html"
m.save(map_filename)
print(f"\nInteractive supply chain map saved as '{map_filename}'.")

# =============================================================================
# PART 3: MCMC Simulation with Animation (Sampling from Standard Normal)
# =============================================================================
def target_density(x):
    return np.exp(-0.5 * x**2)

def run_mcmc(n_iter=200, proposal_std=0.5):
    current = 0.0
    chain = [current]
    for i in range(n_iter):
        candidate = current + np.random.normal(0, proposal_std)
        acceptance_prob = min(1, target_density(candidate) / target_density(current))
        if np.random.rand() < acceptance_prob:
            current = candidate
        chain.append(current)
        print(f"Iteration {i+1:03d}: Sample = {current:.4f}")
    return chain

n_iterations = 200
chain = run_mcmc(n_iter=n_iterations, proposal_std=0.5)

fig_mcmc, (ax_mcmc_trace, ax_mcmc_hist) = plt.subplots(1, 2, figsize=(12, 5))
line_trace, = ax_mcmc_trace.plot([], [], 'b-', lw=2)
# Note: marker data are passed as sequences.
marker_trace, = ax_mcmc_trace.plot([], [], 'ro', markersize=5)
ax_mcmc_trace.set_title("MCMC Chain Trace")
ax_mcmc_trace.set_xlabel("Iteration")
ax_mcmc_trace.set_ylabel("Chain Value")
text_annotation = ax_mcmc_trace.text(0.05, 0.90, "", transform=ax_mcmc_trace.transAxes, fontsize=10, color="darkred")

def init_mcmc():
    ax_mcmc_trace.set_xlim(0, n_iterations)
    ymin = min(chain) - 1
    ymax = max(chain) + 1
    ax_mcmc_trace.set_ylim(ymin, ymax)
    line_trace.set_data([], [])
    marker_trace.set_data([], [])
    ax_mcmc_hist.cla()
    return line_trace, marker_trace, text_annotation

def update_mcmc(frame):
    xdata = list(range(frame+1))
    ydata = chain[:frame+1]
    line_trace.set_data(xdata, ydata)
    marker_trace.set_data([xdata[-1]], [ydata[-1]])  # use list-wrapped values
    text_annotation.set_text(f"Iteration {frame+1}\nValue: {ydata[-1]:.2f}")

    ax_mcmc_hist.cla()
    ax_mcmc_hist.hist(chain[:frame+1], bins=20, density=True, color='gray', alpha=0.7)
    ax_mcmc_hist.set_title("Histogram of Samples")
    ax_mcmc_hist.set_xlabel("Value")
    ax_mcmc_hist.set_ylabel("Density")
    return line_trace, marker_trace, text_annotation

ani_mcmc = animation.FuncAnimation(fig_mcmc, update_mcmc, frames=range(n_iterations),
                                     init_func=init_mcmc, interval=100, repeat=False)
plt.tight_layout()
plt.show()

# =============================================================================
# PART 4: Standard Normal Theoretical Functions, Moments, and Regression Analysis Using Data from Parts 1, 2, and 3
# =============================================================================

# --- Fitted Standard Normal Functions ---
#The fitted functions are based on the Normal distribution fitted to the MCMC Chain Data in Part 5.
#So, we will use the parameters estimated in Part 5

# Assuming mu_hat and std_hat are calculated in Part 5
#mu_hat = 0.1221  # added from part 5.  DEFINED IN PART 5 NOW
#std_hat = 1.0212  # added from part 5. DEFINED IN PART 5 NOW

def fitted_pdf_standard_normal(x, mu, std):
    """Standard normal probability density function."""
    return 1/(std * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu)/std)**2)

def fitted_mgf_standard_normal(t, mu, std):
    """Standard normal moment generating function."""
    return np.exp(mu*t + 0.5*(std*t)**2)



# =============================================================================
# PART 5: Distribution Fitting on MCMC Chain Data
# =============================================================================
# We now use the MCMC simulation chain (which should approximate a standard normal)
# to fit a candidate distribution. In this example, we fit a Normal distribution.

chain_array = np.array(chain)
mu_hat, std_hat = norm.fit(chain_array)  # Fit here, so mu_hat and std_hat are defined.
print("\n--- Distribution Fitting on MCMC Chain Data ---")
print("Fitted Normal Distribution Parameters: mean = {:.4f}, std = {:.4f}".format(mu_hat, std_hat))

# Print the fitted PDF and MGF equations with the fitted parameters.
print("\n--- Fitted Standard Normal Functions ---")
print(f"Fitted PDF: f(x) = 1/({std_hat:.4f}*√(2π)) * exp(-0.5 * ((x-{mu_hat:.4f})/{std_hat:.4f})^2 )")
print(f"Fitted MGF: M(t) = exp({mu_hat:.4f}*t + 0.5*({std_hat:.4f}*t)^2 )") # Corrected this line.

# Q-Q plot for the fitted distribution.
sm.qqplot(chain_array, line='s')
plt.title("Q-Q Plot for Fitted Normal Distribution")
plt.show()

# Plot histogram of the chain data with the fitted PDF overlay.
plt.figure(figsize=(8, 5))
plt.hist(chain_array, bins=30, density=True, alpha=0.6, color='g', label='MCMC Data')
xmin_hist, xmax_hist = plt.xlim()
x_fit = np.linspace(xmin_hist, xmax_hist, 100)
p_fit = norm.pdf(x_fit, mu_hat, std_hat)
plt.plot(x_fit, p_fit, 'k', linewidth=2, label='Fitted Normal PDF')
plt.title("Fitted Distribution to MCMC Chain Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Kolmogorov-Smirnov test for goodness of fit.
ks_stat, p_value = kstest(chain_array, 'norm', args=(mu_hat, std_hat))
print("Kolmogorov-Smirnov Test Statistic: {:.4f}".format(ks_stat))
print("Kolmogorov-Smirnov Test p-value: {:.4f}".format(p_value))

# =============================================================================
# PART 6: Plotting PDF and MGF of Fitted Distribution
# =============================================================================

# Define the fitted PDF and MGF using the estimated parameters.
def fitted_pdf(x):
    return (1 / (std_hat * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_hat) / std_hat) ** 2)

def fitted_mgf(t):
    return np.exp(mu_hat * t + 0.5 * (std_hat * t) ** 2)

# Generate x and t values for plotting.
x_values = np.linspace(chain_array.min() - 1, chain_array.max() + 1, 1000)
t_values = np.linspace(-3, 3, 1000)

# Calculate PDF and MGF values.
pdf_values = fitted_pdf(x_values)
mgf_values = fitted_mgf(t_values)

# Create plots for PDF and MGF.
plt.figure(figsize=(12, 5))

# Plot PDF
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='Fitted PDF', color='blue')
plt.title('Probability Density Function (PDF)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Plot MGF
plt.subplot(1, 2, 2)
plt.plot(t_values, mgf_values, label='Fitted MGF', color='red')
plt.title('Moment Generating Function (MGF)')
plt.xlabel('t')
plt.ylabel('M(t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Animate the Evolution of the Fitted Distribution as More Data Accumulate ---
fig_anim_fit, ax_anim_fit = plt.subplots(figsize=(8, 5))
xmin_fit, xmax_fit = np.min(chain_array)-1, np.max(chain_array)+1
ax_anim_fit.set_xlim(xmin_fit, xmax_fit)
ax_anim_fit.set_ylim(0, 1.2)  # adjust as needed

def init_fit():
    ax_anim_fit.cla()
    ax_anim_fit.set_xlim(xmin_fit, xmax_fit)
    ax_anim_fit.set_ylim(0, 1.2)
    return []

def update_fit(frame):
    ax_anim_fit.cla()
    data_subset = chain_array[:frame+1]
    mu_subset, std_subset = norm.fit(data_subset)
    ax_anim_fit.hist(data_subset, bins=30, density=True, alpha=0.6, color='g')
    x_vals = np.linspace(xmin_fit, xmax_fit, 100)
    y_vals = norm.pdf(x_vals, mu_subset, std_subset)
    ax_anim_fit.plot(x_vals, y_vals, 'k', linewidth=2)
    ax_anim_fit.set_title(f"Frame {frame+1}: Fitted Normal, mu={mu_subset:.2f}, sigma={std_subset:.2f}")
    ax_anim_fit.set_xlabel("Value")
    ax_anim_fit.set_ylabel("Density")
    return []

ani_fit = animation.FuncAnimation(fig_anim_fit, update_fit, frames=range(20, len(chain_array)),
                                     init_func=init_fit, interval=100, repeat=False)
plt.tight_layout()
plt.show()

# =============================================================================
# PART 7: Folium Map with PDF/MGF Route and Distribution Parameters
# =============================================================================
#  Generate points for plotting PDF and MGF on the map
x_values_for_map = np.linspace(chain_array.min() - 2, chain_array.max() + 2, 50)  # More points for smoother curves
pdf_values_for_map = fitted_pdf(x_values_for_map)
t_values_for_map = np.linspace(-4, 4, 50)
mgf_values_for_map = fitted_mgf(t_values_for_map)

# Create a base map centered around the data
mean_lat_map = data['latitude'].mean()
mean_lon_map = data['longitude'].mean()
map_final = folium.Map(location=[mean_lat_map, mean_lon_map], zoom_start=6)

# Add a layer for the PDF route
pdf_coords = list(zip(pdf_values_for_map, x_values_for_map))  # Combine into list of (lat, lon)
folium.PolyLine(locations=pdf_coords, color='blue', weight=2,
                popup='PDF Route').add_to(map_final)

# Add a layer for the MGF route
mgf_coords = list(zip(mgf_values_for_map, t_values_for_map))
folium.PolyLine(locations=mgf_coords, color='red', weight=2,
                popup='MGF Route').add_to(map_final)

# Add a marker for the Mean
folium.Marker(
    location=[fitted_pdf(mu_hat), mu_hat],  # Mean on x-axis, PDF at mean on y-axis
    popup=f'Mean: {mu_hat:.2f}',
    icon=folium.Icon(color='green', icon='info-sign')
).add_to(map_final)

# Add a marker for the Standard Deviation (roughly, one std dev above mean)
folium.Marker(
    location=[fitted_pdf(mu_hat + std_hat), mu_hat + std_hat],
    popup=f'Std Dev: {std_hat:.2f}',
    icon=folium.Icon(color='orange', icon='info-sign')
).add_to(map_final)
# Save the map
map_filename_final = "pdf_mgf_map.html"
map_final.save(map_filename_final)
print(f"\nPDF and MGF routes, mean and standard deviation map saved as '{map_filename_final}'.")

# =============================================================================
# PART 8: Re-do Part 2 with Data from Parts 5 and 6
# =============================================================================

# Use fitted distribution (Part 5/6) to generate supply and demand.  This is the core change.
num_warehouses = 4
num_demands = 6
np.random.seed(42) # Keep it consistent.

# Generate supply and demand values using the fitted normal distribution.
warehouse_supply = np.round(np.random.normal(mu_hat, std_hat, num_warehouses) * 1000).astype(int)  # Scale up
warehouse_supply = np.maximum(warehouse_supply, 100).astype(int) # Ensure supply is positive and not too small.

demand_values = np.round(np.random.normal(mu_hat, std_hat, num_demands) * 500).astype(int) # Scale
demand_values = np.maximum(demand_values, 50).astype(int) # Ensure demand is positive and not too small

# Create new data for Part 2 using the generated supply and demand.
rows_part2 = []

for i in range(num_warehouses):
    node_id = f"W{i+1}"
    node_type = "warehouse"
    latitude = np.random.uniform(30, 50)
    longitude = np.random.uniform(-120, -70)
    supply_value = warehouse_supply[i]
    demand_value = 0
    region = ""
    rows_part2.append([node_id, node_type, latitude, longitude, supply_value, demand_value, region])

regions_part2 = ["NE_Corridor", "Pacific_NW", "Deep_South"]
for i in range(num_demands):
    node_id = f"D{i+1}"
    node_type = "demand"
    latitude = np.random.uniform(25, 50)
    longitude = np.random.uniform(-125, -70)
    supply_value = 0
    demand_value = demand_values[i]
    region = np.random.choice(regions_part2)
    rows_part2.append([node_id, node_type, latitude, longitude, supply_value, demand_value, region])

df_part2 = pd.DataFrame(rows_part2, columns=["node_id", "node_type", "latitude", "longitude", "supply", "demand", "region"])

warehouses_part2 = df_part2[df_part2['node_type'] == 'warehouse'].copy()
demand_nodes_part2 = df_part2[df_part2['node_type'] == 'demand'].copy()

supply_dict_part2 = warehouses_part2.set_index('node_id')['supply'].to_dict()
demand_dict_part2 = demand_nodes_part2.set_index('node_id')['demand'].to_dict()

costs_part2 = {}
for idx, row_w in warehouses_part2.iterrows():
    for jdx, row_d in demand_nodes_part2.iterrows():
        c = haversine(row_w['longitude'], row_w['latitude'],
                        row_d['longitude'], row_d['latitude'])
        costs_part2[(row_w['node_id'], row_d['node_id'])] = c

prob_part2 = pulp.LpProblem("Medicine_Supply_Chain_Optimization_Part2", pulp.LpMinimize)
routes_part2 = pulp.LpVariable.dicts("Shipments_Part2",
                                         (supply_dict_part2.keys(), demand_dict_part2.keys()),
                                         lowBound=0, cat='Continuous')

prob_part2 += pulp.lpSum([routes_part2[i][j] * costs_part2[(i, j)]
                                for i in supply_dict_part2.keys()
                                for j in demand_dict_part2.keys()]), "Total_Transportation_Cost_Part2"

for i in supply_dict_part2.keys():
    prob_part2 += pulp.lpSum([routes_part2[i][j] for j in demand_dict_part2.keys()]) <= supply_dict_part2[i], f"Supply_{i}_Part2"

for j in demand_dict_part2.keys():
    prob_part2 += pulp.lpSum([routes_part2[i][j] for i in supply_dict_part2.keys()]) >= demand_dict_part2[j], f"Demand_{j}_Part2"

prob_part2.solve()

shipment_results_part2 = []
for i in supply_dict_part2.keys():
    for j in demand_dict_part2.keys():
        qty = routes_part2[i][j].varValue
        if qty and qty > 1e-6:
            shipment_results_part2.append((i, j, qty))

print("\nSupply Chain Optimization Results (Part 8, using Fitted Distribution)")
print("Status:", pulp.LpStatus[prob_part2.status])
print("Total Minimum Transportation Cost: {:.2f}".format(pulp.value(prob_part2.objective)))
print("Shipment Plan:")
for (w, d, q) in shipment_results_part2:
    print(f"  Ship from {w} to {d}: {q:.2f} units")

# --- Create Folium Map for Part 8 ---
mean_lat_part2 = df_part2['latitude'].mean()
mean_lon_part2 = df_part2['longitude'].mean()
map_part2 = folium.Map(location=[mean_lat_part2, mean_lon_part2], zoom_start=5)

for idx, row in warehouses_part2.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Warehouse {row['node_id']}<br>Supply: {row['supply']} units<br>Lat: {row['latitude']:.2f}, Lng: {row['longitude']:.2f}",
        icon=folium.Icon(color='blue', icon='building')
    ).add_to(map_part2)

for idx, row in demand_nodes_part2.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"Facility {row['node_id']}<br>Demand: {row['demand']} units<br>"
                f"Region: {row['region']}<br>Lat: {row['latitude']:.2f}, Lng: {row['longitude']:.2f}"),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(map_part2)

for i, j, qty in shipment_results_part2:
    supply_coords = warehouses_part2[warehouses_part2['node_id'] == i][['latitude', 'longitude']].values[0]
    demand_coords = demand_nodes_part2[demand_nodes_part2['node_id'] == j][['latitude', 'longitude']].values[0]
    folium.PolyLine(
        locations=[(supply_coords[0], supply_coords[1]),
                   (demand_coords[0], demand_coords[1])],
        color="green", weight=2, opacity=0.8,
        popup=f"Ship {qty:.2f} units"
    ).add_to(map_part2)

map_filename_part2 = "supply_chain_map_part2.html"
map_part2.save(map_filename_part2)
print(f"\nSupply chain map (Part 8) saved as '{map_filename_part2}'.")

# --- Matplotlib version of the Part 8 map ---
fig_part2, ax_part2 = plt.subplots(figsize=(10, 8))
for idx, row in warehouses_part2.iterrows():
    ax_part2.plot(row['longitude'], row['latitude'], 'bo', label=f"Warehouse {row['node_id']}")
    ax_part2.text(row['longitude'], row['latitude'], f"W{row['node_id']}", fontsize=8, ha='right', va='bottom')

for idx, row in demand_nodes_part2.iterrows():
    ax_part2.plot(row['longitude'], row['latitude'], 'ro', label=f"Demand {row['node_id']}")
    ax_part2.text(row['longitude'], row['latitude'], f"D{row['node_id']}", fontsize=8, ha='left', va='top')

for i, j, qty in shipment_results_part2:
    supply_coords = warehouses_part2[warehouses_part2['node_id'] == i][['latitude', 'longitude']].values[0]
    demand_coords = demand_nodes_part2[demand_nodes_part2['node_id'] == j][['latitude', 'longitude']].values[0]
    ax_part2.plot([supply_coords[1], demand_coords[1]], [supply_coords[0], demand_coords[0]], 'g-', alpha=0.7, label=f'Shipment {qty:.2f}')

ax_part2.set_xlabel('Longitude')
ax_part2.set_ylabel('Latitude')
ax_part2.set_title('Supply Chain Network (Part 8)')
ax_part2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
plt.grid(True)
plt.show()

# --- Folium map for MCMC Simulation (Part 3) ---
# Since MCMC generates a sequence of points, we can represent this as a path on a map
# Assume a hypothetical geographic space for the MCMC samples (for visualization)
mcmc_latitudes = [37 + 2 * np.sin(i * 0.5) for i in range(n_iterations)]  # Example: Sinusoidal variation
mcmc_longitudes = [-95 + 3 * np.cos(i * 0.5) for i in range(n_iterations)]

map_mcmc = folium.Map(location=[np.mean(mcmc_latitudes), np.mean(mcmc_longitudes)], zoom_start=4)

# Add the MCMC path as a PolyLine
mcmc_coords = list(zip(mcmc_latitudes, mcmc_longitudes))
folium.PolyLine(locations=mcmc_coords, color='purple', weight=2,
                popup='MCMC Sample Path').add_to(map_mcmc)

# Add markers for the first and last samples
folium.Marker(location=[mcmc_latitudes[0], mcmc_longitudes[0]],
              popup='Start of MCMC Chain',
              icon=folium.Icon(color='green')).add_to(map_mcmc)
folium.Marker(location=[mcmc_latitudes[-1], mcmc_longitudes[-1]],
              popup='End of MCMC Chain',
              icon=folium.Icon(color='red')).add_to(map_mcmc)

map_filename_mcmc = "mcmc_simulation_map.html"
map_mcmc.save(map_filename_mcmc)
print(f"\nMCMC simulation path map saved as '{map_filename_mcmc}'.")
