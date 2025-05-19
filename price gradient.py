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
    print(f"===== {title.upper()} ======")
    print("="*80)

def load_data(file_path):
    """
    Loads data from an Excel file into a Pandas DataFrame.
    Handles potential errors during file loading.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def display_and_get_input(df, prompt, num_samples=5):
    """
    Displays a sample of the DataFrame and prints its shape.
    Prompts the user for input and handles potential errors.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        prompt (str): The prompt message to display to the user.
        num_samples (int, optional): The number of sample rows to display. Defaults to 5.

    Returns:
        str: User input or None in case of an error.
    """
    if df is None:
        print("Error: DataFrame is None.")
        return None

    try:
        print(prompt)
        print(df.sample(num_samples))
        print(f"Original data shape: {df.shape}")
        user_input = input("Press Enter to continue, or type 'exit' to stop: ").strip().lower()
        return user_input
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_age_column(df):
    """Creates an 'AGE' column from 'YEAR BUILT', handling potential errors."""
    if df is None:
        return None
    try:
        df['AGE'] = 2024 - df['YEAR BUILT']  # Hardcoded 2024.  This should be a parameter, or better, datetime.now().year
        return df
    except KeyError:
        print("Error: 'YEAR BUILT' column not found.")
        return None
    except Exception as e:
        print(f"An error occurred while creating the 'AGE' column: {e}")
        return None

def handle_missing_values(df, columns, strategy='drop'):
    """
    Handles missing values in specified columns using either 'drop' or 'fillna' strategies.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): A list of column names to handle missing values in.
        strategy (str, optional): The strategy to use ('drop' or 'fillna'). Defaults to 'drop'.
            If 'fillna', missing values are filled with the mean of the column.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled, or None in case of an error.
    """
    if df is None:
        return None

    try:
        if strategy == 'drop':
            df.dropna(subset=columns, inplace=True)
            print(f"Dropped rows with missing values in columns: {columns}")
        elif strategy == 'fillna':
            for col in columns:
                df[col].fillna(df[col].mean(), inplace=True)
                print(f"Filled missing values in column '{col}' with its mean.")
        else:
            print(f"Error: Invalid strategy '{strategy}'.  Choose 'drop' or 'fillna'.")
            return None
        return df
    except KeyError as e:
        print(f"Error: Column not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while handling missing values: {e}")
        return None

def convert_to_datetime(df, column):
    """
    Converts a column to datetime format, handling potential errors and invalid dates.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The modified DataFrame, or None in case of an error.
    """
    if df is None:
        return None
    try:
        # errors='coerce' will set invalid dates to NaT (Not a Time)
        df[column] = pd.to_datetime(df[column], errors='coerce')
        # Drop rows where conversion resulted in NaT
        df.dropna(subset=[column], inplace=True)
        print(f"Converted column '{column}' to datetime format.")
        return df
    except KeyError:
        print(f"Error: Column not found: {column}")
        return None
    except Exception as e:
        print(f"An error occurred while converting column '{column}' to datetime: {e}")
        return None

def filter_data(df, condition):
    """
    Filters the DataFrame based on a given condition, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        condition (str): The filtering condition (as a string to be evaluated).

    Returns:
        pd.DataFrame: The filtered DataFrame, or None in case of an error.
    """
    if df is None:
        return None
    try:
        # Use df.query() which is safer and more readable for complex conditions
        filtered_df = df.query(condition).copy() # Ensure we are working on a copy.
        print(f"Filtered data using condition: {condition}")
        return filtered_df
    except Exception as e:
        print(f"An error occurred while filtering data: {e}")
        return None

def perform_regression(df, formula):
    """
    Performs a linear regression using the specified formula, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        formula (str): The regression formula (e.g., 'SALE PRICE ~ GROSS SQUARE FEET + AGE').

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The regression results, or None in case of an error.
    """
    if df is None:
        return None
    try:
        model = sm.OLS.from_formula(formula, data=df)
        results = model.fit()
        print(results.summary())
        return results
    except Exception as e:
        print(f"An error occurred during regression: {e}")
        return None

def calculate_residuals(df, model, x_col, y_col):
    """
    Calculates residuals from a regression model and adds them to the DataFrame.
     Handles cases where the model is None.

    Args:
        df (pd.DataFrame): The DataFrame.
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): The regression model.
        x_col (str): The name of the independent variable column.
        y_col (str): The name of the dependent variable column.

    Returns:
        pd.DataFrame: The DataFrame with residuals added, or None if model is None.
    """
    if df is None or model is None:
        return None
    try:
        predictions = model.predict(df[x_col])
        df['Residuals'] = df[y_col] - predictions
        return df
    except KeyError as e:
        print(f"Error: Column not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while calculating residuals: {e}")
        return None

def plot_residuals(df, x_col, y_col, title='Residuals Plot'):
    """
    Plots residuals against the independent variable, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame containing the data and residuals.
        x_col (str): The name of the independent variable column.
        y_col (str): The name of the dependent variable.
        title (str, optional): The title of the plot.  Defaults to 'Residuals Plot'.
    """
    if df is None:
        print("Error: DataFrame is None. Cannot plot residuals.")
        return
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df['Residuals'], alpha=0.7)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except KeyError as e:
        print(f"Error: Column not found: {e}.  Cannot plot residuals.")
    except Exception as e:
        print(f"An error occurred while plotting residuals: {e}")

def plot_histogram(df, column, title):
    """
    Plots a histogram of a specified column, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame.
        column (str): The name of the column to plot.
        title (str): The title of the plot.
    """
    if df is None:
        print(f"Error: DataFrame is None. Cannot plot histogram for {column}")
        return
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    except KeyError:
        print(f"Error: Column '{column}' not found. Cannot plot histogram.")
    except Exception as e:
        print(f"An error occurred while plotting histogram for {column}: {e}")

def plot_scatter(df, x_col, y_col, title):
    """
    Plots a scatter plot of two columns, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame.
        x_col (str): The name of the x-axis column.
        y_col (str): The name of the y-axis column.
        title (str): The title of the plot.
    """
    if df is None:
        print(f"Error: DataFrame is None. Cannot plot scatter plot for {x_col} and {y_col}")
        return
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.7)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except KeyError:
        print(f"Error: Column(s) not found. Cannot plot scatter plot for {x_col} and {y_col}")
    except Exception as e:
        print(f"An error occurred while plotting scatter plot for {x_col} and {y_col}: {e}")

def calculate_vif(df, features):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in a linear regression model.
    Handles potential errors, including cases where the DataFrame or features are invalid.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list): A list of feature names (column names) to calculate VIF for.

    Returns:
        pd.DataFrame: A DataFrame containing the VIF for each feature, or None in case of an error.
    """
    if df is None or not isinstance(features, list) or not features:
        print("Error: Invalid DataFrame or features list.")
        return None

    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        vif_data["VIF"] = [
            (
                1 / (1 - sm.OLS(df[feature], df.drop(columns=[feature])).fit().rsquared)
                if len(df.columns) > 1  # Avoid error if only one column
                else np.nan  # Return NaN if only one column
            )
            for feature in features
        ]
        return vif_data
    except KeyError as e:
        print(f"Error: Column not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while calculating VIF: {e}")
        return None

def plot_3d_scatter(df, x_col, y_col, z_col, title):
    """
    Plots a 3D scatter plot, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame.
        x_col (str): The name of the x-axis column.
        y_col (str): The name of the y-axis column.
        z_col (str): The name of the z-axis column.
        title (str): The title of the plot.
    """
    if df is None:
        print(f"Error: DataFrame is None. Cannot plot 3D scatter plot.")
        return None
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x_col], df[y_col], df[z_col], c='b', marker='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    except KeyError:
        print(f"Error: Column(s) not found. Cannot plot 3D scatter plot.")
    except Exception as e:
        print(f"An error occurred while plotting 3D scatter plot: {e}")

def plot_covariance_and_correlation(df, numerical_features):
    """
    Plots the covariance and correlation matrices for numerical features, handling potential errors.

    Args:
        df (pd.DataFrame): The DataFrame.
        numerical_features (list):  A list of numerical features.
    """
    if df is None:
        print("Error: DataFrame is None.  Cannot plot covariance and correlation.")
        return

    try:
        # Select only the numerical features and drop rows with any missing values in those features
        numerical_features_cov = df[numerical_features].dropna()

        if not numerical_features_cov.empty and len(numerical_features_cov.columns) > 1:
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
    except KeyError as e:
        print(f"Error: Column not found: {e}")
    except Exception as e:
        print(f"An error occurred while plotting covariance and correlation: {e}")

def geocode_addresses(df):
    """
    Geocodes addresses using Nominatim, handling potential errors and rate limits.

    Args:
        df (pd.DataFrame): The DataFrame containing address information.  It must contain
                         'ADDRESS' and 'ZIP CODE' columns.

    Returns:
        pd.DataFrame: A new DataFrame with 'ADDRESS', 'ZIP CODE', 'LATITUDE', and 'LONGITUDE' columns,
                      or None if geocoding fails or geopy is not available.  Important: returns a *new* dataframe.
    """
    if not GEOPY_AVAILABLE:
        print("Geocoding is disabled because the geopy library is not available.")
        return None

    if df is None:
        print("Error: DataFrame is None. Cannot perform geocoding.")
        return None

    # Create a copy to avoid modifying the original DataFrame in place.
    df_geocoded = df.copy()

    try:
        geolocator = Nominatim(user_agent="nyc_real_estate_analysis")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # Apply rate limiting

        # Create a new 'ADDRESS' column if it doesn't exist.
        if 'ADDRESS' not in df_geocoded.columns:
            df_geocoded['ADDRESS'] = df_geocoded['BLOCK'].astype(str) + ' ' + df_geocoded['LOT'].astype(str)

        # Geocode addresses.  Use a sample for demonstration.
        num_rows_to_geocode = min(100, len(df_geocoded))  # Geocode a maximum of 100 rows
        print(f"--- Starting Geocoding (Nominatim) ---")
        print("IMPORTANT: Nominatim is a free service with rate limits (1 req/sec).")
        print("For large datasets, this will be VERY SLOW and may lead to IP blocks.")
        print("This script will geocode a SAMPLE of the data to demonstrate functionality.")
        print(f"Sampling the first {num_rows_to_geocode} rows for geocoding.")

        tqdm.pandas()  # Initialize tqdm for progress tracking with pandas
        location_results = df_geocoded.head(num_rows_to_geocode).apply(
            lambda row: geocode(f"{row['ADDRESS']}, New York, {row['ZIP CODE']}"), axis=1
        )

        # Create new columns 'LATITUDE' and 'LONGITUDE'
        df_geocoded.loc[:num_rows_to_geocode-1, 'LATITUDE'] = [
            loc.latitude if loc else np.nan for loc in location_results
        ]
        df_geocoded.loc[:num_rows_to_geocode-1, 'LONGITUDE'] = [
            loc.longitude if loc else np.nan for loc in location_results
        ]
        print("Geocoding complete for the sample.")
        return df_geocoded[['ADDRESS', 'ZIP CODE', 'LATITUDE', 'LONGITUDE']] # Return a *new* DataFrame

    except Exception as e:
        print(f"An error occurred during geocoding: {e}")
        return None



# --- 1. Data Loading and Preprocessing ---
file_path = "C:\\Users\\esisy\\Downloads\\rollingsales_brooklyn.xlsx" #hardcoded
df = load_data(file_path)
if df is None:
    exit()  # Stop execution if data loading failed

user_input = display_and_get_input(df, "Here's a sample of the original data:")
if user_input == 'exit':
    exit()

df = create_age_column(df)
if df is None:
    exit()

df = handle_missing_values(df, ['GROSS SQUARE FEET', 'SALE PRICE'], strategy='drop')
if df is None:
    exit()
df = handle_missing_values(df, ['LAND SQUARE FEET'], strategy='fillna') #handle missing values
if df is None:
    exit()

df = convert_to_datetime(df, 'SALE DATE')
if df is None:
    exit()

df = filter_data(df, '`SALE PRICE` > 0 & `GROSS SQUARE FEET` > 0')
if df is None:
    exit()

# --- 2. Feature Engineering (Geocoding) ---
if GEOPY_AVAILABLE:
    # Geocode the data (or a sample of it)
    df_geocoded = geocode_addresses(df)  #  Get the new DataFrame
    if df_geocoded is not None:
        # Merge the geocoded coordinates back into the original DataFrame
        df = pd.merge(df, df_geocoded, on=['ADDRESS', 'ZIP CODE'], how='left')
        print("Merged geocoded data into main DataFrame.")
    else:
        print("Geocoding failed; proceeding without location data.")
else:
    print("Geocoding was skipped because geopy is not installed.")

# --- 3. Regression Analysis ---
print_heading("3. Regression Analysis")
regression_formula = 'SALE PRICE ~ GROSS SQUARE FEET + AGE'
results = perform_regression(df, regression_formula)
if results is not None:
    df = calculate_residuals(df, results, 'GROSS SQUARE FEET', 'SALE PRICE')
    if df is not None:
        plot_residuals(df, 'GROSS SQUARE FEET', 'SALE PRICE')

# --- 4. Further Analysis and Visualization ---
print_heading("4. Further Analysis and Visualization")
plot_histogram(df, 'SALE PRICE', 'Distribution of Sale Price')
plot_scatter(df, 'GROSS SQUARE FEET', 'SALE PRICE', 'Sale Price vs. Gross Square Feet')
plot_3d_scatter(df, 'GROSS SQUARE FEET', 'AGE', 'SALE PRICE', '3D Scatter Plot of Sale Price, Gross Square Feet, and Age')

# VIF Calculation
numerical_features = ['GROSS SQUARE FEET', 'AGE']
vif_df = calculate_vif(df, numerical_features)
if vif_df is not None:
    print(vif_df)

plot_covariance_and_correlation(df, ['SALE PRICE', 'GROSS SQUARE FEET', 'AGE'])

# --- 5. Price Gradient Analysis ---
if GEOPY_AVAILABLE:
    print_heading("5. Price Gradient Analysis")
    #  Symbolic variables
    lat_sym, lon_sym = sympy.symbols('lat lon')

    # 5.1 Define a symbolic price function (replace with your actual function)
    # P_coord_sym = 100000 + 50000 * lat_sym - 30000 * lon_sym #original
    P_coord_sym = 50000*lat_sym - 30000*lon_sym + (1000*lat_sym - 40606.17329375)*(lon_sym + 74.0090673625) - 3250580.6855625

    # 5.2 Calculate the gradient
    grad_P_lat = sympy.diff(P_coord_sym, lat_sym)
    grad_P_lon = sympy.diff(P_coord_sym, lon_sym)

    # 5.3 Calculate the curl and divergence (Laplacian)
    curl_P_coord = sympy.diff(grad_P_lon, lat_sym) - sympy.diff(grad_P_lat, lon_sym)
    laplacian_P_coord = sympy.diff(grad_P_lat, lat_sym) + sympy.diff(grad_P_lon, lon_sym)

    print(f"Assumed Price Function for Coordinate Gradient: P(LAT, LON) = {P_coord_sym}")
    print(f"Gradient of P w.r.t. LATITUDE: ∂P/∂LAT = {grad_P_lat}")
    print(f"Gradient of P w.r.t. LONGITUDE: ∂P/∂LON = {grad_P_lon}")
    print(f"Curl(∇P_coord) = {curl_P_coord} (since P_coord is a scalar potential)")
    print(f"Divergence(∇P_coord) = ∇²P_coord = {laplacian_P_coord}")

    # 5.4 Evaluate the gradient at sample points
    # Use a few sample points from your data for evaluation
    sample_df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).head(5)  # Get first 5 non-NaN locations
    if not sample_df.empty: #check if the dataframe isempty
        lat_sample = sample_df['LATITUDE'].tolist()
        lon_sample = sample_df['LONGITUDE'].tolist()

        grad_lat_func = sympy.lambdify(lat_sym, grad_P_lat, 'numpy')
        grad_lon_func = sympy.lambdify(lon_sym, grad_P_lon, 'numpy')

        grad_lat_values = [grad_lat_func(lat) for lat in lat_sample]
        grad_lon_values = [grad_lon_func(lon) for lon in lon_sample]

        print("\nSample evaluated gradient components at various (LAT, LON) points:")
        print("LAT values:", lat_sample)
        print("LON values:", lon_sample)
        print("∂P/∂LAT values:", grad_lat_values)
        print("∂P/∂LON values:", grad_lon_values)

        # Evaluate Laplacian for plotting
        laplacian_eval = np.array([laplacian_P_coord.evalf(subs={lat_sym: vlat, lon_sym: vlon}) for vlat, vlon in zip(lat_sample, lon_sample)], dtype=float)
        print(f"Sample ∇²P_coord values: {laplacian_eval[:5]}")

        # 5.5 Create a contour plot of the Laplacian (Divergence of the Gradient)
        plt.figure(figsize=(8, 6))
        # Create a grid of latitude and longitude values for the contour plot.  Use the min/max of your data.
        lat_min, lat_max = df['LATITUDE'].min(), df['LATITUDE'].max()
        lon_min, lon_max = df['LONGITUDE'].min(), df['LONGITUDE'].max()
        grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

        laplacian_grid_func = sympy.lambdify((lat_sym, lon_sym), laplacian_P_coord, 'numpy')
        laplacian_grid = laplacian_grid_func(grid_lat, grid_lon)

        # ---  ADD THIS CHECK  ---
        if isinstance(laplacian_grid, (int, float)):  # If Laplacian is a constant
            laplacian_grid_2d = np.full_like(grid_lat, laplacian_grid)  # Make it 2D
        else:
            laplacian_grid_2d = laplacian_grid
        #  ---  END OF ADDED CHECK ---

        plt.contourf(grid_lon, grid_lat, laplacian_grid_2d, levels=20, cmap='coolwarm')
        plt.colorbar(label='Laplacian of Price (∇²P)')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Laplacian of Price Function (Divergence of Gradient)")
        plt.scatter(lon_sample, lat_sample, c=laplacian_eval[:len(lon_sample)], cmap='coolwarm', edgecolors='k')
        plt.show()
    else:
        print("Skipping Price Gradient Analysis:  No valid LATITUDE and LONGITUDE data available.")
else:
    print("Skipping Price Gradient Analysis: geopy is not available.")

print("\nEnd of script.")
