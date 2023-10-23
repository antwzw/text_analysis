import pandas as pd

# Assuming you have already loaded your data into a DataFrame called 'data'.
# If not, replace 'path_to_your_data.csv' with the actual path and load the data.
data = pd.read_csv('/Users/zhengbaoqin/Desktop/fz/ColabteslaAnalysis.csv')

# Get the column names of the DataFrame
column_names = data.columns

# Print the column names
print(column_names)