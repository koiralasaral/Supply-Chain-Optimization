import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split # Not explicitly used in the current flow for a single model fit
# from sklearn.compose import ColumnTransformer # Not explicitly used
# from sklearn.pipeline import Pipeline # Not explicitly used
# from sklearn.metrics import mean_squared_error # Not explicitly used for model evaluation display
import sympy
from sympy.abc import x, y, z, a, b, c # Define some common symbols
import statsmodels.api as sm

# Attempt to import geopy, provide instructions if not found
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    from tqdm import tqdm # For progress bar during geocoding
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("WARNING: geopy library not found. Geocoding will be skipped.")
    print("Please install it by running: pip install geopy tqdm")


# --- 0. Utility Functions ---
def print_heading(title):
    """Prints a formatted heading."""
    print("\n" + "="*80)
    print(f"===== {title.upper()} =====")
    print("="*80 + "\n")

def clean_numerical_column(series):
    """Cleans a numerical column (removes non-numeric, converts to float)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned_series = series.astype(str).str.replace(r'[$,\s]', '', regex=True)
    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
    return numeric_series

# --- 1. Data Loading and Preprocessing ---
print_heading("1. Data Loading and Preprocessing")

file_path = "C:\\Users\\esisy\\Downloads\\rollingsales_brooklyn.xlsx"
try:
    df_brooklyn_orig = pd.read_excel(file_path, skiprows=4)
    print(f"Successfully loaded data from: {file_path}")
except FileNotFoundError:
    print(f"WARNING: File not found at {file_path}.")
    print("Creating a synthetic dataset for demonstration purposes.")
    # (Synthetic data generation code from previous version - kept for fallback)
    data_synthetic = {
        'BOROUGH': ['BROOKLYN'] * 100,
        'NEIGHBORHOOD': np.random.choice(['NeighborhoodA', 'NeighborhoodB', 'NeighborhoodC'], 100),
        'BUILDING CLASS CATEGORY': np.random.choice(['01 ONE FAMILY DWELLINGS', '02 TWO FAMILY DWELLINGS', '03 THREE FAMILY DWELLINGS'], 100),
        'TAX CLASS AT PRESENT': np.random.choice(['1', '2', '2A', '2B'], 100),
        'BLOCK': np.random.randint(100, 1000, 100),
        'LOT': np.random.randint(1, 200, 100),
        'BUILDING CLASS AT PRESENT': np.random.choice(['A1', 'B2', 'C3'], 100),
        'ADDRESS': [f'{i} Main St' for i in range(100)], # Synthetic addresses
        'APARTMENT NUMBER': [None] * 100,
        'ZIP CODE': np.random.randint(11201, 11250, 100), # Synthetic zip codes
        'RESIDENTIAL UNITS': np.random.randint(1, 4, 100),
        'COMMERCIAL UNITS': np.random.randint(0, 2, 100),
        'TOTAL UNITS': np.random.randint(1, 5, 100),
        'LAND SQUARE FEET': np.random.uniform(1000, 5000, 100),
        'GROSS SQUARE FEET': np.random.uniform(1000, 6000, 100),
        'YEAR BUILT': np.random.randint(1900, 2020, 100),
        'TAX CLASS AT TIME OF SALE': np.random.choice([1, 2], 100),
        'BUILDING CLASS AT TIME OF SALE': np.random.choice(['A1', 'B2', 'C3'], 100),
        'SALE PRICE': np.random.uniform(200000, 2000000, 100),
        'SALE DATE': pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), 100))
    }
    df_brooklyn_orig = pd.DataFrame(data_synthetic)
    df_brooklyn_orig['SALE PRICE'] = df_brooklyn_orig['SALE PRICE'].astype(str)
    df_brooklyn_orig['GROSS SQUARE FEET'] = df_brooklyn_orig['GROSS SQUARE FEET'].astype(str)
    df_brooklyn_orig['LAND SQUARE FEET'] = df_brooklyn_orig['LAND SQUARE FEET'].astype(str)

# Make a copy to preserve the original loaded data
df_brooklyn = df_brooklyn_orig.copy()

# Data Cleaning (as before)
print("\nOriginal data sample:")
print(df_brooklyn.head())
print(f"\nOriginal data shape: {df_brooklyn.shape}")

for col in ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET']:
    if col in df_brooklyn.columns:
        df_brooklyn[col] = clean_numerical_column(df_brooklyn[col])

if 'YEAR BUILT' in df_brooklyn.columns:
    df_brooklyn['YEAR BUILT'] = pd.to_numeric(df_brooklyn['YEAR BUILT'], errors='coerce')

current_year = pd.Timestamp.now().year
if 'YEAR BUILT' in df_brooklyn.columns:
    df_brooklyn['AGE'] = current_year - df_brooklyn['YEAR BUILT']
else:
    df_brooklyn['AGE'] = np.random.randint(5, 100, len(df_brooklyn))

# --- Geocoding Step ---
if GEOPY_AVAILABLE:
    print("\n--- Starting Geocoding (Nominatim) ---")
    print("IMPORTANT: Nominatim is a free service with rate limits (1 req/sec).")
    print("For large datasets, this will be VERY SLOW and may lead to IP blocks.")
    print("This script will geocode a SAMPLE of the data to demonstrate functionality.")
    
    # SAMPLING: Geocode only the first N rows. Adjust N or remove sampling for full dataset (at your own risk and time).
    SAMPLE_SIZE_GEOCODING = 100 
    if len(df_brooklyn) > SAMPLE_SIZE_GEOCODING:
        print(f"Sampling the first {SAMPLE_SIZE_GEOCODING} rows for geocoding.")
        df_to_geocode = df_brooklyn.head(SAMPLE_SIZE_GEOCODING).copy()
    else:
        df_to_geocode = df_brooklyn.copy()

    geolocator = Nominatim(user_agent="property_analyzer_script_1.0") # IMPORTANT: Change user_agent
    # RateLimiter ensures we don't hit the server too fast (1 query per second)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10.0, max_retries=2, swallow_exceptions=True)

    latitudes = []
    longitudes = []

    # Ensure 'ADDRESS' and 'ZIP CODE' columns exist
    if 'ADDRESS' not in df_to_geocode.columns:
        print("WARNING: 'ADDRESS' column not found. Skipping geocoding.")
        df_to_geocode['LATITUDE'] = np.nan
        df_to_geocode['LONGITUDE'] = np.nan
    elif 'ZIP CODE' not in df_to_geocode.columns:
        print("WARNING: 'ZIP CODE' column not found. Geocoding may be less accurate.")
        # Proceeding without ZIP, but it's not ideal
        for address_val in tqdm(df_to_geocode['ADDRESS'].astype(str), desc="Geocoding Addresses"):
            location = geocode(address_val, timeout=10)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(np.nan)
                longitudes.append(np.nan)
        df_to_geocode['LATITUDE'] = latitudes
        df_to_geocode['LONGITUDE'] = longitudes
    else:
        # Combine address and zip code for better accuracy
        full_addresses = df_to_geocode['ADDRESS'].astype(str) + ", Brooklyn, NY " + df_to_geocode['ZIP CODE'].astype(str).str.split('.').str[0]
        for address_val in tqdm(full_addresses, desc="Geocoding Addresses"):
            location = geocode(address_val, timeout=10) # timeout in seconds
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(np.nan)
                longitudes.append(np.nan)
        df_to_geocode['LATITUDE'] = latitudes
        df_to_geocode['LONGITUDE'] = longitudes

    # Merge geocoded data back to the original df_brooklyn
    if 'LATITUDE' in df_to_geocode.columns: # Check if geocoding produced columns
        df_brooklyn = df_brooklyn.merge(df_to_geocode[['LATITUDE', 'LONGITUDE']], left_index=True, right_index=True, how='left')
    else: # If geocoding was skipped or failed to add columns
        df_brooklyn['LATITUDE'] = np.nan
        df_brooklyn['LONGITUDE'] = np.nan
        
    print("Geocoding complete for the sample.")
    print(df_brooklyn[['ADDRESS', 'ZIP CODE', 'LATITUDE', 'LONGITUDE']].head())
else: # GEOPY_AVAILABLE is False
    print("Geocoding skipped as geopy library is not available.")
    df_brooklyn['LATITUDE'] = np.nan # Create dummy columns if geopy not available
    df_brooklyn['LONGITUDE'] = np.nan


# Select relevant features for modeling, now including LATITUDE and LONGITUDE
df_analysis_cols = ['SALE PRICE', 'GROSS SQUARE FEET', 'AGE', 'LATITUDE', 'LONGITUDE']

# Add LOCATION_PROXY based on neighborhood if NEIGHBORHOOD column exists
if 'NEIGHBORHOOD' in df_brooklyn.columns:
    neighborhood_prices = df_brooklyn.groupby('NEIGHBORHOOD')['SALE PRICE'].median().sort_values()
    # df_brooklyn might have new NaNs in SALE PRICE if cleaning failed for some rows
    # Ensure neighborhood_prices is not all NaN
    if not neighborhood_prices.empty and not neighborhood_prices.isnull().all():
        median_of_medians = neighborhood_prices.median() # Fallback for neighborhoods not in training or new ones
        
        # Map, ensuring alignment with df_brooklyn's index
        df_brooklyn['LOCATION_PROXY_TEMP'] = df_brooklyn['NEIGHBORHOOD'].map(lambda n: neighborhood_prices.get(n, median_of_medians))
        
        if 'LOCATION_PROXY_TEMP' in df_brooklyn.columns and df_brooklyn['LOCATION_PROXY_TEMP'].std() != 0 and not df_brooklyn['LOCATION_PROXY_TEMP'].isnull().all():
             df_brooklyn['LOCATION_PROXY'] = (df_brooklyn['LOCATION_PROXY_TEMP'] - df_brooklyn['LOCATION_PROXY_TEMP'].mean()) / df_brooklyn['LOCATION_PROXY_TEMP'].std()
        else:
             df_brooklyn['LOCATION_PROXY'] = np.random.randn(len(df_brooklyn)) # Fallback
        df_brooklyn.drop('LOCATION_PROXY_TEMP', axis=1, inplace=True, errors='ignore')
        df_analysis_cols.append('LOCATION_PROXY')
    else:
        df_brooklyn['LOCATION_PROXY'] = np.random.randn(len(df_brooklyn)) # Fallback if neighborhood_prices is problematic
        if 'LOCATION_PROXY' not in df_analysis_cols : df_analysis_cols.append('LOCATION_PROXY') # ensure it's considered
else:
    df_brooklyn['LOCATION_PROXY'] = np.random.randn(len(df_brooklyn)) # Fallback if no NEIGHBORHOOD column
    if 'LOCATION_PROXY' not in df_analysis_cols : df_analysis_cols.append('LOCATION_PROXY')


df_analysis = df_brooklyn[df_analysis_cols].copy()

# Ensure 'SALE PRICE' is numeric before filtering
if 'SALE PRICE' in df_analysis.columns:
    df_analysis['SALE PRICE'] = pd.to_numeric(df_analysis['SALE PRICE'], errors='coerce')

# Handle missing values (e.g., drop rows with NaN in key modeling columns)
# LATITUDE/LONGITUDE might be NaN if geocoding failed or was skipped for some rows
# For analysis requiring coordinates, we'll need to drop rows where they are NaN.
# For now, keep them and let specific sections handle NaNs as needed (e.g. dropna before using coordinates).
df_analysis.dropna(subset=['SALE PRICE', 'GROSS SQUARE FEET', 'AGE'], inplace=True) # Essential columns

if not df_analysis.empty:
    df_analysis = df_analysis[(df_analysis['SALE PRICE'] > 10000) & (df_analysis['GROSS SQUARE FEET'] > 100)]
    df_analysis = df_analysis[df_analysis['AGE'] >= 0]

print("\nCleaned and processed data for analysis (sample):")
print(df_analysis.head())
print(f"\nShape of data for analysis: {df_analysis.shape}")

if df_analysis.empty:
    print("ERROR: No data available after preprocessing. Exiting analysis.")
else:
    print("Proceeding with analysis...")


# --- 2. Calculable Factors and Price Gradient (Differential Calculus) ---
print_heading("2. Calculable Factors and Price Gradient")

# Using Sympy for symbolic differentiation with Latitude and Longitude
area_sym, age_sym, lat_sym, lon_sym = sympy.symbols('area_sym age_sym lat_sym lon_sym')
c0, c1, c2, c3, c4, c5 = sympy.symbols('c0 c1 c2 c3 c4 c5')

# Check if LATITUDE and LONGITUDE columns exist and have non-NaN data
if 'LATITUDE' in df_analysis.columns and 'LONGITUDE' in df_analysis.columns and \
   df_analysis['LATITUDE'].notna().any() and df_analysis['LONGITUDE'].notna().any():
    
    # A hypothetical price function dependent on actual coordinates
    # P_coord_sym = c0 + c1*lat_sym + c2*lon_sym + c3*lat_sym**2 + c4*lon_sym**2 + c5*lat_sym*lon_sym
    # For simplicity in demonstration, using a simpler form:
    # (Note: Real price functions over lat/lon are complex and usually non-polynomial)
    # Coefficients are arbitrary for this symbolic example.
    P_coord_sym = 1000000 + 50000*(lat_sym - df_analysis['LATITUDE'].mean()) \
                         - 30000*(lon_sym - df_analysis['LONGITUDE'].mean()) \
                         + 1000*(lat_sym - df_analysis['LATITUDE'].mean())*(lon_sym - df_analysis['LONGITUDE'].mean())
    
    print(f"\nAssumed Price Function for Coordinate Gradient: P(LAT, LON) = {P_coord_sym}")

    grad_P_lat = sympy.diff(P_coord_sym, lat_sym)
    grad_P_lon = sympy.diff(P_coord_sym, lon_sym)
    print(f"Gradient of P w.r.t. LATITUDE: ∂P/∂LAT = {grad_P_lat}")
    print(f"Gradient of P w.r.t. LONGITUDE: ∂P/∂LON = {grad_P_lon}")

    # Evaluate gradient at some points (using actual LATITUDE, LONGITUDE from the geocoded sample)
    df_coord_analysis = df_analysis.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if not df_coord_analysis.empty:
        lat_vals = df_coord_analysis['LATITUDE'].values
        lon_vals = df_coord_analysis['LONGITUDE'].values
        
        sample_size_gradient = min(20, len(lat_vals))
        if sample_size_gradient > 0:
            sample_indices_gradient = np.random.choice(len(lat_vals), size=sample_size_gradient, replace=False)
            lat_sample = lat_vals[sample_indices_gradient]
            lon_sample = lon_vals[sample_indices_gradient]

            grad_lat_eval = np.array([grad_P_lat.evalf(subs={lat_sym: vlat, lon_sym: vlon}) for vlat, vlon in zip(lat_sample, lon_sample)], dtype=float)
            grad_lon_eval = np.array([grad_P_lon.evalf(subs={lat_sym: vlat, lon_sym: vlon}) for vlat, vlon in zip(lat_sample, lon_sample)], dtype=float)

            print(f"\nSample evaluated gradient components at various (LAT, LON) points:")
            print(f"LAT values: {lat_sample[:5]}")
            print(f"LON values: {lon_sample[:5]}")
            print(f"∂P/∂LAT values: {grad_lat_eval[:5]}")
            print(f"∂P/∂LON values: {grad_lon_eval[:5]}")

            # Plotting the price gradient (Quiver plot)
            plt.figure(figsize=(10, 8))
            grid_lat_min, grid_lat_max = lat_vals.min(), lat_vals.max()
            grid_lon_min, grid_lon_max = lon_vals.min(), lon_vals.max()

            # Ensure min != max for linspace
            if grid_lat_min < grid_lat_max and grid_lon_min < grid_lon_max:
                grid_lat, grid_lon = np.meshgrid(np.linspace(grid_lat_min, grid_lat_max, 20),
                                                 np.linspace(grid_lon_min, grid_lon_max, 20))
                P_grid_func = sympy.lambdify((lat_sym, lon_sym), P_coord_sym, 'numpy')
                P_grid = P_grid_func(grid_lat, grid_lon)
                
                plt.contourf(grid_lon, grid_lat, P_grid, levels=20, cmap='viridis', alpha=0.7) # X-axis is Longitude
                plt.colorbar(label='Estimated Price P(LAT, LON)')
                # Adjust quiver scale based on gradient magnitudes, not price grid
                grad_magnitude = np.sqrt(grad_lat_eval**2 + grad_lon_eval**2)
                quiver_scale_val = np.percentile(grad_magnitude, 80) * 10 if len(grad_magnitude)>0 and np.percentile(grad_magnitude, 80)>0 else 1.0
                
                plt.quiver(lon_sample, lat_sample, grad_lon_eval, grad_lat_eval, color='red', scale=quiver_scale_val, headwidth=4, width=0.003)
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title("Price Gradient (∇P) in Coordinate Space")
                plt.grid(True)
                plt.show()

                laplacian_P_coord = sympy.diff(grad_P_lat, lat_sym) + sympy.diff(grad_P_lon, lon_sym)
                print(f"\nCurl(∇P_coord) = 0 (since P_coord is a scalar potential)")
                print(f"Divergence(∇P_coord) = ∇²P_coord = {laplacian_P_coord}")
                laplacian_eval = np.array([laplacian_P_coord.evalf(subs={lat_sym: vlat, lon_sym: vlon}) for vlat, vlon in zip(lat_sample, lon_sample)], dtype=float)
                print(f"Sample ∇²P_coord values: {laplacian_eval[:5]}")
                
                plt.figure(figsize=(8, 6))
                laplacian_grid_func = sympy.lambdify((lat_sym, lon_sym), laplacian_P_coord, 'numpy')
                laplacian_grid = laplacian_grid_func(grid_lat, grid_lon)
                plt.contourf(grid_lon, grid_lat, laplacian_grid, levels=20, cmap='coolwarm')
                plt.colorbar(label='Laplacian of Price (∇²P)')
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title("Laplacian of Price Function (Divergence of Gradient)")
                plt.scatter(lon_sample, lat_sample, c=laplacian_eval[:len(lon_sample)], cmap='coolwarm', edgecolors='k')
                plt.show()
            else:
                print("Could not create grid for gradient plot (latitude/longitude range too small or identical).")
        else:
            print("Not enough geocoded data points for gradient visualization sampling.")
    else:
        print("No valid geocoded (Latitude, Longitude) data available in df_analysis for gradient calculation.")
else:
    print("LATITUDE and LONGITUDE columns not available or all NaN. Skipping coordinate-based gradient calculation and plots.")


# --- 3. Integral Calculus ---
print_heading("3. Integral Calculus")
if 'LATITUDE' in df_analysis.columns and 'LONGITUDE' in df_analysis.columns and \
   df_analysis['LATITUDE'].notna().any() and df_analysis['LONGITUDE'].notna().any() and \
   'P_coord_sym' in locals():
    
    df_coord_integral = df_analysis.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if not df_coord_integral.empty:
        lat_min_val, lat_max_val = df_coord_integral['LATITUDE'].min(), df_coord_integral['LATITUDE'].max()
        lon_min_val, lon_max_val = df_coord_integral['LONGITUDE'].min(), df_coord_integral['LONGITUDE'].max()
        
        if lat_max_val > lat_min_val and lon_max_val > lon_min_val:
            # Symbolic integration over lat, then lon
            integral_P_dlat = sympy.integrate(P_coord_sym, (lat_sym, lat_min_val, lat_max_val))
            integral_P_dlatdlon = sympy.integrate(integral_P_dlat, (lon_sym, lon_min_val, lon_max_val))
            
            print(f"\nSymbolic integral of P_coord_sym over LAT from {lat_min_val:.4f} to {lat_max_val:.4f}: {integral_P_dlat}")
            print(f"Symbolic integral of P_coord_sym over LON from {lon_min_val:.4f} to {lon_max_val:.4f} (after integrating over LAT): {integral_P_dlatdlon}")

            # Area in degrees squared (conceptual, not actual geographic area)
            region_deg_area = (lat_max_val - lat_min_val) * (lon_max_val - lon_min_val)
            if region_deg_area > 1e-9: # Avoid division by zero for very small areas
                avg_price_integrated = integral_P_dlatdlon / region_deg_area
                print(f"Total 'value' (∫∫ P dlat dlon) over region [{lat_min_val:.4f}-{lat_max_val:.4f}, {lon_min_val:.4f}-{lon_max_val:.4f}]: {avg_price_integrated*region_deg_area:.2f}")
                print(f"Area of region (degrees squared): {region_deg_area:.6f}")
                print(f"Average price from symbolic integration over the coordinate region: {avg_price_integrated:.2f}")
            else:
                print("Conceptual region area in degrees is too small, cannot calculate average price via integration.")
        else:
            print("Cannot define a valid coordinate region for integration (min/max issue).")
    else:
        print("Not enough valid geocoded data for integral calculus example.")
else:
    print("LATITUDE, LONGITUDE not available, or P_coord_sym not defined. Skipping integral calculus part.")

print("\nFitting data to a differential equation is model-specific and advanced.")
print("Example concept: Price diffusion ∂P/∂t = D * ∇²P or growth dP/dt = rP(1-P/K).")


# --- 4. Sequence and Series ---
print_heading("4. Sequence and Series")
A_param, k_param = 100000, 0.0005
price_exp_func_area_sym = A_param * sympy.exp(k_param * area_sym)
print(f"\nExample exponential price function P(Area) = {price_exp_func_area_sym}")

taylor_exp_area = price_exp_func_area_sym.series(area_sym, x0=0, n=4)
print(f"Taylor series expansion of P(Area) around Area=0 (up to 3rd order):")
print(taylor_exp_area)


# --- 5. Fitting Price Function to Mathematical Forms & Conversions ---
print_heading("5. Fitting Price Function to Mathematical Forms")

# For fitting, we need features that are mostly non-NaN.
# Let's use 'GROSS SQUARE FEET', 'AGE', and 'LONGITUDE' (if available and valid).
# If 'LONGITUDE' is not good, fall back to 'LOCATION_PROXY' or a temp feature.

fitting_features = ['GROSS SQUARE FEET', 'AGE']
loc_feature_for_fit = None

if 'LONGITUDE' in df_analysis.columns and df_analysis['LONGITUDE'].notna().sum() > len(df_analysis) * 0.5: # Require at least 50% non-NaN
    loc_feature_for_fit = 'LONGITUDE'
elif 'LOCATION_PROXY' in df_analysis.columns and df_analysis['LOCATION_PROXY'].notna().sum() > len(df_analysis) * 0.5:
    loc_feature_for_fit = 'LOCATION_PROXY'
    print("Using LOCATION_PROXY for fitting as LONGITUDE has too many NaNs or is unavailable.")
else:
    df_analysis['TEMP_FIT_LOC'] = np.random.rand(len(df_analysis))
    loc_feature_for_fit = 'TEMP_FIT_LOC'
    print("Warning: Using temporary random feature for fitting as primary location features have too many NaNs.")

if loc_feature_for_fit:
    fitting_features.append(loc_feature_for_fit)

# Prepare data for fitting
df_fit_analysis = df_analysis[fitting_features + ['SALE PRICE']].copy()
df_fit_analysis.dropna(subset=fitting_features + ['SALE PRICE'], inplace=True) # Drop rows with NaNs in selected features or target

if df_fit_analysis.empty or len(df_fit_analysis) < 10:
    print("Insufficient data for fitting price functions after NaN handling. Skipping this section.")
else:
    X_fit_cols = fitting_features
    X_fit = df_fit_analysis[X_fit_cols]
    y_fit = df_fit_analysis['SALE PRICE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_fit.columns, index=X_fit.index)

    print("\nFeatures for fitting (scaled, sample):")
    print(X_scaled_df.head())
    
    # For plotting P vs Area, keeping Age and the location feature constant at their means
    area_range_scaled = np.linspace(X_scaled_df['GROSS SQUARE FEET'].min(), X_scaled_df['GROSS SQUARE FEET'].max(), 100)
    age_mean_scaled_fit = X_scaled_df['AGE'].mean()
    loc_mean_scaled_fit = X_scaled_df[loc_feature_for_fit].mean() if loc_feature_for_fit in X_scaled_df.columns else 0

    # 5.1. Polynomial Function
    print("\n--- 5.1 Polynomial Function Fit ---")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled_df)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y_fit)
    print(f"Polynomial Regression (degree 2) R-squared: {poly_reg.score(X_poly, y_fit):.4f}")
    # print(f"Coefficients: {poly_reg.coef_}") # Can be very long
    # print(f"Intercept: {poly_reg.intercept_}")

    plot_data_poly_area = np.zeros((len(area_range_scaled), X_scaled_df.shape[1]))
    plot_data_poly_area[:,0] = area_range_scaled
    plot_data_poly_area[:,1] = age_mean_scaled_fit
    if X_scaled_df.shape[1] > 2: # If location feature is included
        plot_data_poly_area[:,2] = loc_mean_scaled_fit
    X_plot_poly_area_transformed = poly.transform(plot_data_poly_area)
    price_poly_area = poly_reg.predict(X_plot_poly_area_transformed)

    # 5.2. Exponential Function
    print("\n--- 5.2 Exponential Function Fit ---")
    y_log = np.log(y_fit.replace(0, 1e-9))
    exp_reg = LinearRegression()
    exp_reg.fit(X_scaled_df, y_log)
    print(f"Exponential Regression (Linear fit on log(Price)) R-squared (on log(Price)): {exp_reg.score(X_scaled_df, y_log):.4f}")

    plot_data_exp_area = np.zeros((len(area_range_scaled), X_scaled_df.shape[1]))
    plot_data_exp_area[:,0] = area_range_scaled
    plot_data_exp_area[:,1] = age_mean_scaled_fit
    if X_scaled_df.shape[1] > 2:
        plot_data_exp_area[:,2] = loc_mean_scaled_fit
    log_price_exp_area = exp_reg.predict(plot_data_exp_area)
    price_exp_area = np.exp(log_price_exp_area)

    # 5.3. Trigonometric Function (Simplified)
    print("\n--- 5.3 Trigonometric Function Fit (Conceptual/Simplified) ---")
    def trig_func_simple(area_scaled, A, B, C, D):
        return A + B * np.sin(C * area_scaled + D)
    x_area_scaled_simple = X_scaled_df['GROSS SQUARE FEET'].values
    y_price_simple = y_fit.values
    try:
        initial_guesses = [y_price_simple.mean(), y_price_simple.std(), 1, 0]
        params_trig, _ = curve_fit(trig_func_simple, x_area_scaled_simple, y_price_simple, p0=initial_guesses, maxfev=10000, bounds=([-np.inf, -np.inf, -5, -np.pi], [np.inf, np.inf, 5, np.pi]))
        A_trig, B_trig, C_trig, D_trig = params_trig
        print(f"Fitted Trigonometric parameters (simple P(Area)): A={A_trig:.2f}, B={B_trig:.2f}, C={C_trig:.2f}, D={D_trig:.2f}")
        price_trig_area = trig_func_simple(area_range_scaled, *params_trig)
    except RuntimeError:
        print("Could not fit trigonometric function with curve_fit. Using a placeholder.")
        A_trig, B_trig, C_trig, D_trig = y_price_simple.mean(), 0, 1, 0 # Placeholder with B=0 if fit fails
        price_trig_area = trig_func_simple(area_range_scaled, A_trig, B_trig, C_trig, D_trig)

    # 5.4. Conversions (Taylor Series Demo)
    print("\n--- 5.4 Conversions between Forms (Taylor Series Demo) ---")
    k_taylor = sympy.Symbol('k_taylor')
    # Using sympy.abc.x for the symbolic variable in Taylor series
    exp_form_sym = sympy.exp(k_taylor * sympy.abc.x) 
    poly_from_exp_sym = exp_form_sym.series(sympy.abc.x, 0, 4)
    print(f"Taylor expansion of exp(k*x) around x=0: {poly_from_exp_sym}")
    sin_form_sym = sympy.sin(k_taylor * sympy.abc.x)
    poly_from_sin_sym = sin_form_sym.series(sympy.abc.x, 0, 4)
    print(f"Taylor expansion of sin(k*x) around x=0: {poly_from_sin_sym}")
    
    # 5.5. Plotting Fitted Functions (Price vs. Area)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_scaled_df['GROSS SQUARE FEET'], y_fit, alpha=0.3, label="Actual Data (Scaled Area)")
    plt.plot(area_range_scaled, price_poly_area, color='red', linestyle='--', linewidth=2, label="Polynomial Fit")
    plt.plot(area_range_scaled, price_exp_area, color='green', linestyle=':', linewidth=2, label="Exponential Fit")
    plt.plot(area_range_scaled, price_trig_area, color='purple', linestyle='-.', linewidth=2, label="Trigonometric Fit (Simple P(Area))")
    plt.xlabel("Scaled Gross Square Feet")
    plt.ylabel("Sale Price")
    plt.title("Fitted Price Functions vs. Scaled Area (Other factors at mean)")
    plt.legend()
    plt.ylim(0, np.percentile(y_fit, 99.5) * 1.1 if len(y_fit) > 0 else 1000000)
    plt.grid(True)
    plt.show()

    # 3D Plot for Polynomial Fit (Price vs Area and Age, Loc feature constant)
    if X_scaled_df.shape[1] >= 2: # Need at least Area and Age
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        area_scaled_grid_3d = np.linspace(X_scaled_df['GROSS SQUARE FEET'].min(), X_scaled_df['GROSS SQUARE FEET'].max(), 30)
        age_scaled_grid_3d = np.linspace(X_scaled_df['AGE'].min(), X_scaled_df['AGE'].max(), 30)
        area_mesh, age_mesh = np.meshgrid(area_scaled_grid_3d, age_scaled_grid_3d)
        
        # Prepare features for prediction, keeping the location feature at its mean
        X_plot_3d_features_list = [area_mesh.ravel(), age_mesh.ravel()]
        if X_scaled_df.shape[1] > 2 and loc_feature_for_fit in X_scaled_df.columns: # If location feature was used
            loc_mesh_3d = np.full(area_mesh.shape, X_scaled_df[loc_feature_for_fit].mean())
            X_plot_3d_features_list.append(loc_mesh_3d.ravel())
        
        X_plot_3d_features = np.c_[tuple(X_plot_3d_features_list)]
        
        X_plot_3d_poly = poly.transform(X_plot_3d_features)
        price_poly_3d_surface = poly_reg.predict(X_plot_3d_poly).reshape(area_mesh.shape)
        
        ax.plot_surface(area_mesh, age_mesh, price_poly_3d_surface, cmap='viridis', alpha=0.8)
        
        sample_size_3d = min(200, len(X_scaled_df))
        if sample_size_3d > 0:
            sample_idx_3d = np.random.choice(X_scaled_df.index, size=sample_size_3d, replace=False)
            ax.scatter(X_scaled_df.loc[sample_idx_3d, 'GROSS SQUARE FEET'], 
                       X_scaled_df.loc[sample_idx_3d, 'AGE'], 
                       y_fit.loc[sample_idx_3d], color='red', s=10, label="Actual Data")

        ax.set_xlabel("Scaled Gross Square Feet")
        ax.set_ylabel("Scaled Age")
        ax.set_zlabel("Sale Price")
        ax.set_title(f"3D Poly Fit: Price vs. Area & Age ({loc_feature_for_fit} at mean)")
        plt.show()


# --- 6. Mathematical Analysis (Real and Complex Analysis) ---
print_heading("6. Mathematical Analysis of Fitted Functions")
if 'P_coord_sym' in locals() and 'grad_P_lat' in locals() and 'grad_P_lon' in locals():
    print(f"\nAnalyzing P_coord_sym = {P_coord_sym}")
    try:
        # lat_sym, lon_sym are the sympy symbols
        critical_points_coord = sympy.solve([grad_P_lat, grad_P_lon], (lat_sym, lon_sym))
        print(f"Critical points of P_coord_sym: {critical_points_coord}")
        
        Hlatlat = sympy.diff(P_coord_sym, lat_sym, 2)
        Hlonlon = sympy.diff(P_coord_sym, lon_sym, 2)
        Hlatlon = sympy.diff(P_coord_sym, lat_sym, lon_sym)
        Determinant_H_coord = Hlatlat * Hlonlon - Hlatlon**2
        print(f"H_latlat = {Hlatlat}, H_lonlon = {Hlonlon}, H_latlon = {Hlatlon}")
        print(f"Determinant of Hessian D_coord = {Determinant_H_coord}")

        points_to_check_coord = []
        if isinstance(critical_points_coord, dict):
            points_to_check_coord.append(critical_points_coord)
        elif isinstance(critical_points_coord, list):
            for item in critical_points_coord:
                if isinstance(item, tuple) and len(item) == 2:
                     points_to_check_coord.append({lat_sym: item[0], lon_sym: item[1]})
                elif isinstance(item, dict):
                     points_to_check_coord.append(item)
        
        if not points_to_check_coord and not (Determinant_H_coord.is_number and Determinant_H_coord == 0):
            print("No specific critical points found by sympy.solve() for P_coord_sym.")
            # Further checks if gradient is constant etc.
            if grad_P_lat.is_number and grad_P_lon.is_number and grad_P_lat == 0 and grad_P_lon == 0:
                print("Gradient is zero everywhere; function might be constant.")
            elif grad_P_lat.is_number and grad_P_lon.is_number:
                 print("Gradient is constant but non-zero; function is linear, no critical points of type max/min/saddle.")

        for crit_pt_dict in points_to_check_coord:
            eval_subs_coord = {k: (float(v) if hasattr(v, 'is_Number') and v.is_Number else v) for k, v in crit_pt_dict.items()}
            D_val_c = Determinant_H_coord.evalf(subs=eval_subs_coord) if not Determinant_H_coord.is_number else Determinant_H_coord
            Hll_val_c = Hlatlat.evalf(subs=eval_subs_coord) if not Hlatlat.is_number else Hlatlat
            P_val_c = P_coord_sym.evalf(subs=eval_subs_coord)
            
            D_val_num_c = float(D_val_c) if hasattr(D_val_c, 'is_Number') and D_val_c.is_Number else D_val_c
            Hll_val_num_c = float(Hll_val_c) if hasattr(Hll_val_c, 'is_Number') and Hll_val_c.is_Number else Hll_val_c

            print(f"At critical point {crit_pt_dict}:")
            print(f"  P_value = {P_val_c}")
            print(f"  D_coord = {D_val_num_c}, H_latlat = {Hll_val_num_c}")
            if isinstance(D_val_num_c, (int, float)) and isinstance(Hll_val_num_c, (int, float)):
                if D_val_num_c > 0 and Hll_val_num_c > 0: print("  Nature: Local Minimum")
                elif D_val_num_c > 0 and Hll_val_num_c < 0: print("  Nature: Local Maximum")
                elif D_val_num_c < 0: print("  Nature: Saddle Point")
                else: print("  Nature: Test inconclusive")
            else:
                print("  Nature: Could not determine numerically.")
    except Exception as e:
        print(f"Could not solve for critical points of P_coord_sym: {e}")
else:
    print("P_coord_sym or its gradients not defined. Skipping mathematical analysis of critical points for coordinates.")
print("\nComplex analysis is typically not directly applied unless for specific theoretical models.")


# --- 7. Statistics ---
print_heading("7. Statistics")

if df_analysis.empty or len(df_analysis) < 2:
    print("Insufficient data for statistical analysis. Skipping this section.")
else:
    price_data = df_analysis['SALE PRICE'].dropna()
    if price_data.empty or len(price_data) < 2:
        print("Not enough price data after dropna for statistical analysis.")
    else:
        print("\n--- 7.1 Expectation (Mean) ---")
        mean_price = price_data.mean()
        mean_area = df_analysis['GROSS SQUARE FEET'].dropna().mean()
        mean_age = df_analysis['AGE'].dropna().mean()
        print(f"E[Price] = {mean_price:,.2f}")
        print(f"E[Area] = {mean_area:,.2f} sq ft")
        print(f"E[Age] = {mean_age:,.2f} years")
        if 'LATITUDE' in df_analysis.columns and df_analysis['LATITUDE'].notna().any():
            print(f"E[Latitude] = {df_analysis['LATITUDE'].dropna().mean():.4f}")
        if 'LONGITUDE' in df_analysis.columns and df_analysis['LONGITUDE'].notna().any():
            print(f"E[Longitude] = {df_analysis['LONGITUDE'].dropna().mean():.4f}")


        print("\n--- 7.2 Moments (for Price) ---")
        # (Moments calculation as before)
        m1_origin = price_data.mean()
        m2_origin = (price_data**2).mean()
        print(f"1st moment about origin (mean): {m1_origin:,.2f}")
        print(f"2nd moment about origin: {m2_origin:,.2f}")
        var_pop = price_data.var(ddof=0) 
        std_pop = np.sqrt(var_pop) if var_pop >= 0 else 0
        m1_mean = (price_data - mean_price).mean() 
        m2_mean = var_pop
        m3_mean = stats.skew(price_data, bias=True) * (std_pop**3) if std_pop > 0 else 0
        m4_mean = stats.kurtosis(price_data, fisher=False, bias=True) * (std_pop**4) if std_pop > 0 else 0
        print(f"\n1st central moment (should be ~0): {m1_mean:,.2f}")
        print(f"2nd central moment (variance, ddof=0): {m2_mean:,.2f}")
        print(f"3rd central moment: {m3_mean:,.2f}")
        print(f"4th central moment: {m4_mean:,.2f}")
        print(f"Standard Deviation of Price (ddof=1): {price_data.std():,.2f}")
        print(f"Skewness of Price: {stats.skew(price_data):.4f}") 
        print(f"Kurtosis of Price (Fisher, excess): {stats.kurtosis(price_data, fisher=True):.4f}")


        print("\n--- 7.3 Moment Generating Functions (Conceptual) ---")
        print("MGF depends on the chosen probability distribution.")

        print("\n--- 7.4 Probability Distribution Fitting (for Price) ---")
        plt.figure(figsize=(10, 6))
        sns.histplot(price_data, kde=True, stat="density", label="Price Distribution", bins=50)
        try:
            shape_ln, loc_ln, scale_ln = stats.lognorm.fit(price_data, floc=0)
            fitted_lognorm = stats.lognorm(s=shape_ln, loc=loc_ln, scale=scale_ln)
            x_plot_dist = np.linspace(price_data.min(), price_data.max(), 200)
            plt.plot(x_plot_dist, fitted_lognorm.pdf(x_plot_dist), 'r-', lw=2, label=f'Fitted Log-Normal\n(s={shape_ln:.2f}, scale={scale_ln:,.0f})')
            print(f"Fitted Log-Normal parameters: shape={shape_ln:.2f}, loc={loc_ln:.2f}, scale={scale_ln:,.2f}")
            ks_stat, ks_pvalue = stats.kstest(price_data, lambda x_val: fitted_lognorm.cdf(x_val))
            print(f"Kolmogorov-Smirnov test for Log-Normal fit: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
            if ks_pvalue < 0.05: print("  KS test suggests data may not follow Log-Normal (p < 0.05).")
            else: print("  KS test does not reject Log-Normal hypothesis (p >= 0.05).")
        except Exception as e:
            print(f"Could not fit log-normal distribution: {e}")
        plt.title("Sale Price Distribution and Fitted Log-Normal")
        plt.xlabel("Sale Price")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("\n--- 7.5 Regression Analysis (Detailed) ---")
        # Use X_scaled_df and y_fit from Section 5 which are already cleaned and aligned
        if 'X_scaled_df' in locals() and isinstance(X_scaled_df, pd.DataFrame) and not X_scaled_df.empty and \
           'y_fit' in locals() and isinstance(y_fit, pd.Series) and not y_fit.empty:
            if X_scaled_df.index.equals(y_fit.index): # Double check alignment
                y_sm = y_fit
                X_sm = sm.add_constant(X_scaled_df.copy()) # Use a copy to avoid SettingWithCopyWarning if X_scaled_df is used later
                
                model_sm = sm.OLS(y_sm, X_sm)
                results_sm = model_sm.fit()
                print(results_sm.summary())
                
                plt.figure(figsize=(10, 6))
                residuals = results_sm.resid
                plt.scatter(results_sm.fittedvalues, residuals, alpha=0.5)
                plt.axhline(0, color='red', linestyle='--')
                plt.xlabel("Fitted Values")
                plt.ylabel("Residuals")
                plt.title("Residual Plot from OLS Regression")
                plt.grid(True)
                plt.show()

                fig_qq = sm.qqplot(residuals, line='s')
                plt.title("Q-Q Plot of Residuals")
                plt.show()
            else:
                print("Skipping statsmodels regression: X_scaled_df and y_fit indices (from Sec 5) are misaligned.")
        else:
            print("Skipping statsmodels regression as input data (X_scaled_df or y_fit from Sec 5) is not prepared.")

        print("\n--- 7.6 Covariance Matrix ---")
        cols_for_cov = ['SALE PRICE', 'GROSS SQUARE FEET', 'AGE']
        if 'LATITUDE' in df_analysis.columns and df_analysis['LATITUDE'].notna().any():
            cols_for_cov.append('LATITUDE')
        if 'LONGITUDE' in df_analysis.columns and df_analysis['LONGITUDE'].notna().any():
            cols_for_cov.append('LONGITUDE')
        if 'LOCATION_PROXY' in df_analysis.columns and df_analysis['LOCATION_PROXY'].notna().any():
             cols_for_cov.append('LOCATION_PROXY')
        
        cols_for_cov = [col for col in cols_for_cov if col in df_analysis.columns] # Ensure columns exist
        
        if len(cols_for_cov) > 1:
            numerical_features_cov = df_analysis[cols_for_cov].copy()
            numerical_features_cov.dropna(inplace=True)

            if not numerical_features_cov.empty and len(numerical_features_cov.columns) > 1 and len(numerical_features_cov) > 1:
                cov_matrix = numerical_features_cov.cov()
                print("Covariance Matrix:")
                print(cov_matrix)
                plt.figure(figsize=(10, 7))
                sns.heatmap(cov_matrix, annot=True, fmt=".2e", cmap="coolwarm", annot_kws={"size": 8})
                plt.title("Covariance Matrix of Key Features")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
                
                corr_matrix = numerical_features_cov.corr()
                print("\nCorrelation Matrix:")
                print(corr_matrix)
                plt.figure(figsize=(10, 7))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 8})
                plt.title("Correlation Matrix of Key Features")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
            else:
                print("Not enough numerical features or data after dropna for covariance/correlation matrix.")
        else:
            print("Not enough valid columns selected for covariance/correlation matrix.")

print("\n" + "="*80)
print("===== ANALYSIS SCRIPT COMPLETE =====")
print("="*80 + "\n")
