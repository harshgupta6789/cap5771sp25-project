import pandas as pd
import os

# Base directory (ensures correct file paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, 'clean.csv')
output_path = os.path.join(BASE_DIR, 'processed.csv')

# Load full dataset
df = pd.read_csv(input_path)

# Compute airline-level metrics
airline_stats = df.groupby('Airline_Code').agg(
    Avg_Airline_Planes_Age=('Avg_Airline_Planes_Age', 'mean'),
    AvgDepDelayByAirline_min=('AvgDepDelayByAirline_min', 'mean'),
    Airplane_Cost_millions=('Airplane_Cost_millions', 'mean')
).reset_index()

# Compute route-level metrics
route_stats = df.groupby('Route').agg(
    OriginState=('OriginState', 'first'),
    DestState=('DestState', 'first'),
    AvgDepDelayByRoute_min=('AvgDepDelayByRoute_min', 'mean'),
    RouteTraffic=('RouteTraffic', 'mean')
).reset_index()

# Get example flight-level values per airline+route
temp = df.groupby(['Airline_Code', 'Route']).agg(
    Scheduled_Elapsed_Time_min=('Scheduled_Elapsed_Time_min', 'first'),
    AirTime_min=('AirTime_min', 'first'),
    Distance_miles=('Distance_miles', 'first'),
    Fare=('Fare', 'first'),
    Origin_City=('Origin_City', 'first'),
    Dest_City=('Dest_City', 'first')
).reset_index()

# Unique airline-route combinations
df_routes = df[['Airline_Code', 'Route']].drop_duplicates()

# Merge all pieces
df_proc = (
    df_routes
    .merge(airline_stats, on='Airline_Code', how='left')
    .merge(route_stats, on='Route', how='left')
    .merge(temp, on=['Airline_Code', 'Route'], how='left')
)

# Save processed CSV
# This contains exactly the columns your Streamlit app references directly from df:
# Airline_Code, Route, Avg_Airline_Planes_Age, AvgDepDelayByAirline_min, Airplane_Cost_millions,
# OriginState, DestState, AvgDepDelayByRoute_min, RouteTraffic,
# Scheduled_Elapsed_Time_min, AirTime_min, Distance_miles, Fare, Origin_City, Dest_City

df_proc.to_csv(output_path, index=False)
print(f"Processed data written to {output_path}")
