import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


folder = "video/_20251015_154007768-15fps-x20RTL/"
# folder = "video/20241022_172649156_5ul_20glic-x20RTL.mov"
# folder = "video/20241023_174413659_5ul_0glic-x20RTL.mov"


xl = pd.ExcelFile( f'{folder}merged.xlsx' )
dfc = xl.parse(xl.sheet_names[0])

df = dfc[ dfc['f'] > 500 ]

print( df.columns )


# Obs graph
plt.figure(figsize=(12, 6))
plt.scatter(df['f'], df['x_px_auto'], label='automatic detection', marker='.', alpha=.6 )
plt.scatter(df['f'], df['x_px_manual'], label='manual detection', marker='.', alpha=.5 )

plt.xlabel('Frame number')
plt.ylabel('Displacement in px')
plt.title('Manual vs automatic detection of movements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{folder}_compare.jpg')
#plt.show()



# --- Quadratic regression ---
mask_auto = ~df['x_px_auto'].isnull()
x_auto = df.loc[mask_auto, 'f'].values.reshape(-1, 1)
y_auto = df.loc[mask_auto, 'x_px_auto'].values

poly = PolynomialFeatures(degree=2)
x_auto_poly = poly.fit_transform(x_auto)
reg_auto = LinearRegression()
reg_auto.fit(x_auto_poly, y_auto)

# Forecast across the entire visible range
xx_auto = np.linspace(x_auto.min(), x_auto.max(), 500).reshape(-1, 1)
yy_auto = reg_auto.predict(poly.transform(xx_auto))
plt.plot(xx_auto, yy_auto, color='blue', linestyle='--', label='Quadratic regression for x_px_auto')

# --- Quadratic regression ---
mask_manual = ~df['x_px_manual'].isnull()
x_manual = df.loc[mask_manual, 'f'].values.reshape(-1, 1)
y_manual = df.loc[mask_manual, 'x_px_manual'].values

x_manual_poly = poly.fit_transform(x_manual)
reg_manual = LinearRegression()
reg_manual.fit(x_manual_poly, y_manual)
xx_manual = np.linspace(x_manual.min(), x_manual.max(), 500).reshape(-1, 1)
yy_manual = reg_manual.predict(poly.transform(xx_manual))
plt.plot(xx_manual, yy_manual, color='orange', linestyle='--', label='Quadratic regression for x_px_manual')

plt.xlabel('Frame')
plt.ylabel('Value position (px)')
plt.title('Quadratic Regression of Displacement (automated vs manual obs.)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{folder}_quadratic_regression.jpg')
#plt.show()




print("\n...............................\n")

# Align frames where both values exist
mask_both = (~df['x_px_auto'].isnull()) & (~df['x_px_manual'].isnull())
auto_values = df.loc[mask_both, 'x_px_auto'].values
manual_values = df.loc[mask_both, 'x_px_manual'].values

rmse_auto_vs_manual = np.sqrt(mean_squared_error(manual_values, auto_values))
print(f"RMSE between x_px_auto and x_px_manual (only valid points in both): {rmse_auto_vs_manual:.4f} px")


rmse = np.sqrt(mean_squared_error(manual_values, auto_values))
range_value = manual_values.max() - manual_values.min()  # or np.mean(manual_values)
rrmse = rmse / range_value * 100

print(f"RRMSE vcompared to range_value (max-min): {rrmse:.2f}%") # RRMSE (Relative RMSE) = ( RMSE / range ) * 100

mean_value = auto_values.mean()
rrmse_mean = rmse / mean_value * 100    # compared to the average

print(f"RRMSE compared to average: {rrmse_mean:.2f}%")

median_value = np.median( auto_values )
rrmse_median = rmse / median_value * 100    # compared to the median

print(f"RRMSE compared to median: {rrmse_median:.2f}%")


# write RMSE file
error_text = f"""
Project folder: {folder} \n\n
RMSE between x_px_auto and x_px_manual (only valid points in both): {rmse_auto_vs_manual:.4f} px 
RRMSE vcompared to range_value (max-min): {rrmse:.2f}% 
RRMSE compared to average: {rrmse_mean:.2f}% 
RRMSE compared to median: {rrmse_median:.2f}% 
"""

with open( f"{folder}/error.txt" , "w", encoding="utf-8") as f:
    f.write( error_text )   # overwrites file or creates it if missing
