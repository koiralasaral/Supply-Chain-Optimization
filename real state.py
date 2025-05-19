import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.interpolate import griddata
from scipy.integrate import simpson
import scipy.stats as stats
import statsmodels.api as sm
import sympy as sp

# -----------------------------
# STEP 1: Load and clean data
# -----------------------------
data_file = r"C:\Users\esisy\Downloads\rollingsales_brooklyn.xlsx"

# Read the Excel file (make sure this file exists at the given location)
df = pd.read_excel(data_file)

# Convert sale price to numeric (remove commas etc.)
if 'SALE PRICE' in df.columns:
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].replace({',': ''}, regex=True), errors='coerce')

# Remove rows with missing or zero sale price
df = df[df['SALE PRICE'] > 0].copy()
df.reset_index(drop=True, inplace=True)

print("=== Data Sample ===")
print(df.head())  # print first few rows as a sample

# ----------------------------------------------------------
# STEP 2: Create synthetic coordinates (if not provided)
# ----------------------------------------------------------
if not ('x' in df.columns and 'y' in df.columns):
    # For demonstration, we create synthetic coordinates roughly in the Brooklyn area.
    np.random.seed(42)
    df['x'] = np.random.uniform(40.5, 40.95, len(df))
    df['y'] = np.random.uniform(-74.2, -73.7, len(df))

# ----------------------------------------------------------
# STEP 3: Build a scalar field from sale price via interpolation
# ----------------------------------------------------------
# Define a grid over the geographic region
xi = np.linspace(df['x'].min(), df['x'].max(), 100)
yi = np.linspace(df['y'].min(), df['y'].max(), 100)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Interpolate the sale prices onto the grid (using cubic interpolation)
Z_grid = griddata((df['x'], df['y']), df['SALE PRICE'], (X_grid, Y_grid), method='cubic')

# In case of NaNs, fill them using nearest-neighbor interpolation
mask = np.isnan(Z_grid)
if np.any(mask):
    Z_grid[mask] = griddata((df['x'], df['y']), df['SALE PRICE'], (X_grid, Y_grid), method='nearest')[mask]

# ----------------------------------------------------------
# STEP 4: Compute Differential Operators on the Price Field
# ----------------------------------------------------------
# Compute spatial step sizes
dx = xi[1]-xi[0]
dy = yi[1]-yi[0]

# Use np.gradient: note that the first returned gradient corresponds to the y-direction 
grad_y, grad_x = np.gradient(Z_grid, dy, dx)

# For a 2D vector field F = (grad_x, grad_y), we compute:
# Curl: (∂grad_y/∂x - ∂grad_x/∂y) (the only nonzero component in 3D)
dgrad_y_dx = np.gradient(grad_y, dx, axis=1)
dgrad_x_dy = np.gradient(grad_x, dy, axis=0)
curl_z = dgrad_y_dx - dgrad_x_dy

# Divergence (for gradient, this is the Laplacian): d(grad_x)/dx + d(grad_y)/dy
dgrad_x_dx = np.gradient(grad_x, dx, axis=1)
dgrad_y_dy = np.gradient(grad_y, dy, axis=0)
divergence = dgrad_x_dx + dgrad_y_dy

# Print some intermediate values at the center index for inspection
center_i, center_j = 50, 50
print("\n=== Differential Calculus on Price Field ===")
print(f"Gradient at center: grad_x = {grad_x[center_i, center_j]:.2f}, grad_y = {grad_y[center_i, center_j]:.2f}")
print(f"Curl (z-component) at center (should be near 0): {curl_z[center_i, center_j]:.2e}")
print(f"Divergence (Laplacian) at center: {divergence[center_i, center_j]:.2f}")

# ----------------------------------------------------------
# STEP 5: Plotting 2D and 3D for the price field & gradient
# ----------------------------------------------------------
# 2D contour with gradient vectors (quiver)
plt.figure(figsize=(10, 8))
contour = plt.contourf(X_grid, Y_grid, Z_grid, 50, cmap='viridis')
plt.colorbar(contour, label='Sale Price')
plt.quiver(X_grid[::5, ::5], Y_grid[::5, ::5], grad_x[::5, ::5], grad_y[::5, ::5], color='white')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Contour of Sale Price with Gradient Vectors')
plt.show()

# 3D Surface plot of sale price
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')
fig.colorbar(surf, ax=ax, label='Sale Price')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Sale Price')
ax.set_title('3D Surface Plot of Sale Price')
plt.show()

# ----------------------------------------------------------
# STEP 6: Integral Calculus - Compute the double integral over the region
# ----------------------------------------------------------
# Here we numerically integrate the sale price field over the grid.
integral_y = simps(Z_grid, dy, axis=0)
integral_total = simps(integral_y, dx)
print("\nDouble integral (numerical) of sale price over the region =", integral_total)

# ----------------------------------------------------------
# STEP 7: Differential Equation and Sequence/Series Examples
# ----------------------------------------------------------
# (A) Differential Equation: Exponential growth dp/dt = r * p.  
r = 0.05   # growth rate
p0 = 100   # initial price index (arbitrary unit)
t = np.linspace(0, 100, 200)
p = p0 * np.exp(r * t)
plt.figure(figsize=(8, 6))
plt.plot(t, p, label=r'$p(t)=p_0\exp(rt)$')
plt.xlabel('Time')
plt.ylabel('Price Index')
plt.title('Exponential Growth (Differential Equation)')
plt.legend()
plt.show()

# 3D visualization: Vary r from 0.02 to 0.08 and show p(t, r)
r_vals = np.linspace(0.02, 0.08, 50)
T, R_vals = np.meshgrid(t, r_vals)
P = p0 * np.exp(R_vals * T)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, R_vals, P, cmap='plasma')
ax.set_xlabel('Time (t)')
ax.set_ylabel('Growth Rate (r)')
ax.set_zlabel('Price Index')
ax.set_title('3D Plot of Exponential Growth for Varying r')
plt.show()

# (B) Sequence and Series: Partial sums of a geometric series a*r^n
n = np.arange(0, 50)
a = 1
r_seq = 0.9
terms = a * r_seq ** n
partial_sums = np.cumsum(terms)
plt.figure(figsize=(8, 6))
plt.plot(n, partial_sums, marker='o')
plt.xlabel('n')
plt.ylabel('Partial Sum')
plt.title('Partial Sums of a Geometric Series (2D)')
plt.grid(True)
plt.show()

# 3D bar plot of the individual series terms
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar(n, terms, zs=0, zdir='y', alpha=0.8)
ax.set_xlabel('n')
ax.set_ylabel('Index')
ax.set_zlabel('Term Value')
ax.set_title('3D Bar Plot of Geometric Series Terms')
plt.show()

# ----------------------------------------------------------
# STEP 8: Mathematical Analysis (Real & Complex)
# ----------------------------------------------------------
# (A) Real Analysis: Use sympy to symbolically differentiate/integrate f(x)=sin(x)*exp(x)
x_sym = sp.symbols('x')
f_sym = sp.sin(x_sym) * sp.exp(x_sym)
f_prime_sym = sp.diff(f_sym, x_sym)
f_int_sym = sp.integrate(f_sym, x_sym)
print("\n=== Symbolic Mathematical Analysis ===")
print("f(x) =", f_sym)
print("f'(x) =", f_prime_sym)
print("Integral of f(x) dx =", f_int_sym)

# Evaluate numerically and plot f(x) and f'(x)
f_numeric = sp.lambdify(x_sym, f_sym, 'numpy')
f_prime_numeric = sp.lambdify(x_sym, f_prime_sym, 'numpy')
xi_vals = np.linspace(0, 2*np.pi, 400)
plt.figure(figsize=(8, 6))
plt.plot(xi_vals, f_numeric(xi_vals), label='f(x)')
plt.plot(xi_vals, f_prime_numeric(xi_vals), label="f'(x)")
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Function f(x)=sin(x)exp(x) and its Derivative')
plt.legend()
plt.show()

# (B) Complex Analysis Demonstration
# Plot the magnitude of f(z)=exp(z) for z = x + iy
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X_comp, Y_comp = np.meshgrid(x_vals, y_vals)
Z_comp = X_comp + 1j * Y_comp
F_comp = np.exp(Z_comp)
F_mag = np.abs(F_comp)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_comp, Y_comp, F_mag, cmap='coolwarm')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_zlabel('|exp(z)|')
ax.set_title('3D Surface Plot of |exp(z)|')
fig.colorbar(surf, ax=ax)
plt.show()

# ----------------------------------------------------------
# STEP 9: Statistics: Expectation, Moments, MGF, Fit a Distribution, Covariance
# ----------------------------------------------------------
sale_prices = df['SALE PRICE'].values
mean_price = np.mean(sale_prices)
variance_price = np.var(sale_prices)
skew_price = stats.skew(sale_prices)
kurtosis_price = stats.kurtosis(sale_prices)
print("\n=== Statistical Analysis for Sale Price ===")
print("Expectation (Mean) =", mean_price)
print("Variance =", variance_price)
print("Skewness =", skew_price)
print("Kurtosis =", kurtosis_price)

# Compute a sample moment generating function (MGF) at a few small t values
t_values = np.linspace(-0.001, 0.001, 5)
mgf_values = [np.mean(np.exp(t * sale_prices)) for t in t_values]
for t_val, mgf_val in zip(t_values, mgf_values):
    print(f"MGF at t={t_val:.5f}: {mgf_val}")

# Fit a normal distribution to the sale prices
mu_fit, sigma_fit = stats.norm.fit(sale_prices)
print("Fitted Normal Distribution: mu =", mu_fit, "sigma =", sigma_fit)

# 2D: Plot histogram of sale prices with fitted Normal PDF
plt.figure(figsize=(8, 6))
count, bins, _ = plt.hist(sale_prices, bins=30, density=True, alpha=0.6, label='Data Histogram')
pdf_fitted = stats.norm.pdf(bins, mu_fit, sigma_fit)
plt.plot(bins, pdf_fitted, 'r-', label='Fitted Normal PDF')
plt.xlabel('Sale Price')
plt.ylabel('Density')
plt.title('Histogram of Sale Prices with Normal PDF Fit')
plt.legend()
plt.show()

# 3D: 3D bar plot of the histogram
hist, bin_edges = np.histogram(sale_prices, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar(bin_centers, hist, zs=0, zdir='y', alpha=0.7)
ax.set_xlabel('Sale Price')
ax.set_ylabel('Frequency Index')
ax.set_zlabel('Density')
ax.set_title('3D Bar Plot of Sale Price Histogram')
plt.show()

# Compute the covariance matrix between x, y and sale price
cov_matrix = np.cov(df[['x', 'y', 'SALE PRICE']].T)
print("\nCovariance Matrix:\n", cov_matrix)

# Display covariance matrix as a table using DataFrame
cov_df = pd.DataFrame(cov_matrix, index=['x', 'y', 'SALE PRICE'], columns=['x', 'y', 'SALE PRICE'])
print("\nCovariance Matrix (DataFrame):\n", cov_df)

# ----------------------------------------------------------
# STEP 10: Regression Analysis and 3D Visualization of the Regression Plane
# ----------------------------------------------------------
# Regress sale price on the coordinates x and y
X_reg = df[['x', 'y']]
X_reg = sm.add_constant(X_reg)  # add intercept term
y_reg = df['SALE PRICE']
model = sm.OLS(y_reg, X_reg).fit()
print("\n=== Regression Analysis Summary ===")
print(model.summary())

# Create predicted regression surface using grid coordinates
predicted_surface = model.params['const'] + model.params['x'] * X_grid + model.params['y'] * Y_grid

# 3D Scatter plot of data with regression plane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['SALE PRICE'], c='b', marker='o', alpha=0.5, label='Data')
ax.plot_surface(X_grid, Y_grid, predicted_surface, color='red', alpha=0.3, label='Regression Plane')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Sale Price')
ax.set_title('3D Scatter Plot with Regression Plane')
plt.show()

# ----------------------------------------------------------
# STEP 11: Linear Algebra: Eigen Decomposition and SVD
# ----------------------------------------------------------
# Eigen decomposition of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("\nEigenvalues of the covariance matrix:", eig_vals)
print("Eigenvectors of the covariance matrix:\n", eig_vecs)

# Plot the eigenvalues in 2D
plt.figure(figsize=(8, 6))
plt.bar(['eig1', 'eig2', 'eig3'], eig_vals)
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of the Covariance Matrix')
plt.show()

# SVD of the regression design matrix (including the constant)
U, s, Vt = np.linalg.svd(X_reg, full_matrices=False)
print("Singular Values of the Regression Design Matrix:", s)

# Plot the singular values using a stem plot
plt.figure(figsize=(8, 6))
plt.stem(s, basefmt=" ")
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Values of Regression Design Matrix')
plt.show()