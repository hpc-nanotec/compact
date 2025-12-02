import pandas as pd


folder = "video/20251015_154007768-15fps-x20RTL/"
# folder = "video/20241022_172649156_5ul_20glic-x20RTL.mov"
# folder = "video/20241023_174413659_5ul_0glic-x20RTL.mov"

# Read datasets
df1 = pd.read_csv(f'{folder}automated.csv')
df2 = pd.read_csv(f'{folder}manual.csv')

df1['x_px_auto'] = df1['x_px']
df1['y_px_auto'] = df1['y_px']
df1['dist_px_auto'] = df1['dist_px']

df1 = df1.drop("x_px", axis='columns')
df1 = df1.drop("y_px", axis='columns')
df1 = df1.drop("dist_px", axis='columns')

df1['x_mm_auto'] = df1['x_mm']
df1['y_mm_auto'] = df1['y_mm']
df1['dist_mm_auto'] = df1['dist_mm']

df1 = df1.drop("x_mm", axis='columns')
df1 = df1.drop("y_mm", axis='columns')
df1 = df1.drop("dist_mm", axis='columns')

print( df1.columns )
print( df2.columns )

# merge dataset with columb: 'frame'
df_merged = pd.merge(df1, df2, on='f', how='outer')



# View and save dataset merged
print( df_merged )

df_merged.to_csv(f'{folder}merged.csv', index=False)

