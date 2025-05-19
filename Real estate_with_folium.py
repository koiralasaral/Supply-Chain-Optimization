import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, HTML
import math
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  # 50 MB

# ======================
# 1. Data Generation
# ======================

np.random.seed(42)
n_parcels = 500

# Generate random locations and areas.
latitudes = 27.7 + np.random.rand(n_parcels) * 0.5      # Approximate latitude in Dhading area
longitudes = 84.8 + np.random.rand(n_parcels) * 0.4      # Approximate longitude in Dhading area
areas = np.random.lognormal(mean=2, sigma=0.5, size=n_parcels)  # Land area in ropani (skewed distribution)

# "Distance" feature from a central point (27.9, 84.9)
center_lat, center_lon = 27.9, 84.9
distances = np.sqrt((latitudes - center_lat)**2 + (longitudes - center_lon)**2)

# Price simulation: price influenced by area and an inverse function of distance, plus noise.
prices = 500000 * areas + 1000000 * (1 / (np.abs(latitudes - 27.9) + 0.1)) + np.random.normal(0, 1e5, n_parcels)

# Organize data into a DataFrame.
df = pd.DataFrame({
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Area': areas,
    'Distance': distances,
    'Price': prices
})

# ======================
# 2. Statistical Analysis & Multicollinearity Check
# ======================

# Compute correlation matrix and Variance Inflation Factor (VIF)
corr_matrix = df[['Latitude', 'Longitude', 'Area', 'Distance']].corr()
vif_data = pd.DataFrame()
vif_data["Feature"] = df[['Latitude', 'Longitude', 'Area', 'Distance']].columns
vif_data["VIF"] = [
    variance_inflation_factor(df[['Latitude', 'Longitude', 'Area', 'Distance']].values, i)
    for i in range(4)
]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# ======================
# 3. Ridge Regression Analysis
# ======================

X = df[['Latitude', 'Longitude', 'Area', 'Distance']]
y = df['Price']

# Standardize features for numerical stability.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge_model = Ridge(alpha=1.0).fit(X_scaled, y)
print("\nRidge Regression Coefficients:")
print(f"Latitude: {ridge_model.coef_[0]:.2f}")
print(f"Longitude: {ridge_model.coef_[1]:.2f}")
print(f"Area: {ridge_model.coef_[2]:.2f}")
print(f"Distance: {ridge_model.coef_[3]:.2f}")
print(f"R² Score: {ridge_model.score(X_scaled, y):.2f}")

# ======================
# 4. Monte Carlo Market Simulation (with Sale Profit Recording)
# ======================

class LandMarket:
    def __init__(self, prices, areas):
        self.assets = []             # Stores bought assets as (purchase_price, area, parcel_index)
        self.cash_history = []       # Tracks cash balance over time
        self.sale_history = []       # Records sale events as (parcel_index, profit, sale_value)
        self.cash = 1e8              # Starting cash amount
        self.price_history = prices.copy()  # Original per-unit prices for each parcel
        self.area_history = areas.copy()    # Areas in ropani for each parcel
        self.profit_history = []     # Records profit from each sale transaction

    def step(self):
        # Random transaction: 60% chance to buy, 40% chance to sell (if assets exist)
        transaction_type = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])
        if transaction_type == 'sell' and self.assets:
            # Sell an asset.
            asset_idx = np.random.randint(len(self.assets))
            purchase_price, area, parcel_idx = self.assets.pop(asset_idx)
            # Compute a fluctuated current per-unit price.
            current_price = self.price_history[parcel_idx] * (0.9 + 0.2 * np.random.rand())
            # Total sale value (sale price multiplied by area).
            sale_value = current_price * area
            # Total cost at purchase.
            cost_value = purchase_price * area
            # Profit (or loss) is the difference.
            profit = sale_value - cost_value
            self.cash += sale_value
            self.profit_history.append(profit)
            self.sale_history.append((parcel_idx, profit, sale_value))
        else:
            # Buy: randomly select a parcel.
            idx = np.random.randint(len(self.price_history))
            price_value = self.price_history[idx] * (0.9 + 0.2 * np.random.rand())
            cost = price_value * self.area_history[idx]
            if self.cash >= cost:
                self.assets.append((price_value, self.area_history[idx], idx))
                self.cash -= cost
        self.cash_history.append(self.cash)

market = LandMarket(df['Price'].values, df['Area'].values)

# ======================
# 5. Animated Visualization of Profit Evolution (Matplotlib)
# ======================

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1000)
ax.set_ylim(-5e6, 5e6)
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    market.step()
    x_data = list(range(len(market.profit_history)))
    y_data = np.cumsum(market.profit_history) if len(market.profit_history) > 0 else [0]
    line.set_data(x_data, y_data)
    ax.set_xlim(0, max(1000, len(x_data) + 100))
    ax.set_title(f"Cumulative Profit: {y_data[-1]:.2f} NPR")
    return line,

ani = FuncAnimation(fig, update, frames=1000, init_func=init, blit=True, interval=50)
display(HTML(ani.to_jshtml()))

# ======================
# 6. Function to Create a Land Parcel Polygon (Simulated as a Square)
# ======================

def get_land_polygon(lat, lon, area):
    """
    Given the center (lat, lon) and land area in ropani, return polygon coordinates.
    1 ropani ≈ 508.72 m². We simulate the parcel as a square polygon centered at (lat, lon).
    """
    SQM_per_ropani = 508.72
    area_sqm = area * SQM_per_ropani
    side_m = np.sqrt(area_sqm)
    
    # Convert side length from meters to degrees (approx. 111111 m per degree latitude)
    delta_lat = (side_m / 111111.0) / 2.0
    # Adjust for longitude (depends on cosine of latitude)
    delta_lon = (side_m / (111111.0 * np.cos(np.radians(lat)))) / 2.0

    polygon = [
        [lat - delta_lat, lon - delta_lon],
        [lat - delta_lat, lon + delta_lon],
        [lat + delta_lat, lon + delta_lon],
        [lat + delta_lat, lon - delta_lon],
        [lat - delta_lat, lon - delta_lon]  # Close polygon
    ]
    return polygon

# ======================
# 7. Interactive Folium Map Visualization with Land Parcel Polygons and Sale Markers
# ======================

# Create a Folium map.
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
parcel_cluster = MarkerCluster(name="Parcels").add_to(m)

for idx, row in df.iterrows():
    poly_coords = get_land_polygon(row['Latitude'], row['Longitude'], row['Area'])
    popup_text = (f"Price: {row['Price']/1e6:.2f}M NPR<br>"
                  f"Area: {row['Area']:.2f} Ropani<br>"
                  f"Distance: {row['Distance']:.3f}°")
    folium.Polygon(
        locations=poly_coords,
        color='blue',
        weight=2,
        fill=True,
        fill_color='blue',
        fill_opacity=0.4,
        popup=popup_text
    ).add_to(parcel_cluster)

sale_cluster = MarkerCluster(name="Sales").add_to(m)
for sale in market.sale_history:
    parcel_idx, profit, sale_value = sale
    row = df.iloc[parcel_idx]
    # Green marker for profit, red for loss.
    marker_color = 'green' if profit > 0 else 'red'
    sale_popup = (f"Sale Event<br>Profit: {profit:.2f} NPR<br>Sale Value: {sale_value:.2f} NPR")
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=sale_popup,
        icon=folium.Icon(color=marker_color, icon='info-sign')
    ).add_to(sale_cluster)

# After simulation, calculate cumulative profit details.
cumulative_profit = np.sum(market.profit_history)
print("\nSale Details:")
for i, sale in enumerate(market.sale_history, start=1):
    parcel_idx, profit, sale_value = sale
    print(f"Sale {i}: Parcel {parcel_idx}, Profit: {profit:.2f} NPR, Sale Value: {sale_value:.2f} NPR")
print(f"Final Cumulative Profit: {cumulative_profit:.2f} NPR")

# Add a marker on the map with cumulative profit details.
folium.Marker(
    location=[center_lat, center_lon],
    icon=folium.DivIcon(html=f"""
        <div style="font-size: 16pt; font-weight: bold; color: black;
                    background-color: white; padding: 5px; border: 1px solid black;">
            Final Cumulative Profit: {cumulative_profit:.2f} NPR
        </div>""")
).add_to(m)

folium.LayerControl().add_to(m)
m.save("advanced_land_market_map.html")
print("Interactive map saved as 'advanced_land_market_map.html'.")

# ======================
# 8. Matplotlib Visualization Equivalent to the Folium Map
#     a) 2D Plot: Land Parcels as Polygons (Latitude vs. Longitude)
#     b) 3D Plot: Scatter Plot with Latitude (x), Longitude (y), and Price (z)
# ======================

from matplotlib.patches import Polygon as mplPolygon

# 8a. 2D Plot of Land Parcels
fig2, ax2 = plt.subplots(figsize=(10, 8))
for idx, row in df.iterrows():
    poly_coords = get_land_polygon(row['Latitude'], row['Longitude'], row['Area'])
    # Color parcels red if price > median, otherwise blue.
    facecolor = 'red' if row['Price'] > df['Price'].median() else 'blue'
    poly_patch = mplPolygon(poly_coords, closed=True, edgecolor='black',
                            facecolor=facecolor, alpha=0.4)
    ax2.add_patch(poly_patch)

margin = 0.05
ax2.set_xlim(df['Latitude'].min() - margin, df['Latitude'].max() + margin)
ax2.set_ylim(df['Longitude'].min() - margin, df['Longitude'].max() + margin)
ax2.set_xlabel('Latitude')
ax2.set_ylabel('Longitude')
ax2.set_title('2D Visualization of Land Parcels')
plt.grid(True)

# 8b. 3D Scatter Plot: Latitude, Longitude, Price
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting.
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
sc = ax3.scatter(df['Latitude'], df['Longitude'], df['Price'], 
                 c=df['Price'], cmap='viridis', marker='o')
ax3.set_xlabel('Latitude')
ax3.set_ylabel('Longitude')
ax3.set_zlabel('Price (NPR)')
ax3.set_title('3D Scatter Plot: Price vs. Latitude and Longitude')
fig3.colorbar(sc, ax=ax3, shrink=0.5, aspect=5)

plt.show()