import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Load data
bikes_df = pd.read_csv('exports/city_summaries_hourly.csv')
weather_df = pd.read_csv('weather_data.csv')

# Convert timestamp columns to datetime
bikes_df['timestamp'] = pd.to_datetime(bikes_df['timestamp'])
# Weather data is in UTC, convert to local time (UTC+2 for CEST, UTC+1 for CET)
weather_df['date'] = pd.to_datetime(weather_df['date'], utc=True)
# Convert UTC to local time (assuming Europe/Berlin timezone)
weather_df['date'] = weather_df['date'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

# Rename weather column for clarity
weather_df.rename(columns={'date': 'timestamp'}, inplace=True)

# Merge data on timestamp
merged_df = pd.merge_asof(
    bikes_df.sort_values('timestamp'),
    weather_df.sort_values('timestamp'),
    on='timestamp',
    direction='nearest'
)

# Group by timestamp and aggregate (in case of multiple cities)
merged_df = merged_df.groupby('timestamp').agg({
    'booked_bikes': 'mean',
    'total_bikes': 'mean',
    'available_bikes': 'mean',
    'temperature_2m': 'first',
    'rain': 'first',
    'cloud_cover': 'first',
    'wind_speed_10m': 'first',
    'relative_humidity_2m': 'first'
}).reset_index()

# Sort by timestamp
merged_df = merged_df.sort_values('timestamp')

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Bike Rentals & Weather Analysis', fontsize=16, fontweight='bold')

# Plot 1: Booked bikes and total bikes
ax1 = axes[0]
ax1.plot(merged_df['timestamp'], merged_df['booked_bikes'], 
         label='Booked Bikes', linewidth=2, color='#FF6B6B', marker='o', markersize=4, alpha=0.8)
ax1.plot(merged_df['timestamp'], merged_df['total_bikes'], 
         label='Total Bikes', linewidth=2, color='#4ECDC4', marker='s', markersize=4, alpha=0.8)
ax1.fill_between(merged_df['timestamp'], merged_df['booked_bikes'], alpha=0.3, color='#FF6B6B')
ax1.set_ylabel('Number of Bikes', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('Rented & Available Bikes Over Time', fontsize=12, fontweight='bold')

# Plot 2: Temperature and Rain with Booked Bikes overlay
ax2 = axes[1]
color_temp = '#FF9800'
ax2.plot(merged_df['timestamp'], merged_df['temperature_2m'], 
         label='Temperature (°C)', linewidth=2.5, color=color_temp, marker='o', markersize=4)
ax2.fill_between(merged_df['timestamp'], merged_df['temperature_2m'], alpha=0.2, color=color_temp)
ax2.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold', color=color_temp)
ax2.tick_params(axis='y', labelcolor=color_temp)

# Add rain as bar chart on secondary axis
ax2_rain = ax2.twinx()
ax2_rain.bar(merged_df['timestamp'], merged_df['rain'], 
             label='Rain (mm)', alpha=0.4, color='#2196F3', width=0.03)
ax2_rain.set_ylabel('Rain (mm)', fontsize=11, fontweight='bold', color='#2196F3')
ax2_rain.tick_params(axis='y', labelcolor='#2196F3')

# Add booked bikes on tertiary axis with proper scaling
ax2_bikes = ax2.twinx()
ax2_bikes.spines['right'].set_position(('outward', 60))
ax2_bikes.plot(merged_df['timestamp'], merged_df['booked_bikes'], 
               label='Booked Bikes', linewidth=2.5, color='#FF6B6B', marker='D', markersize=3, alpha=0.7, linestyle='--')
ax2_bikes.set_ylabel('Booked Bikes', fontsize=11, fontweight='bold', color='#FF6B6B')
ax2_bikes.tick_params(axis='y', labelcolor='#FF6B6B')

ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_title('Temperature, Rain & Booked Bikes', fontsize=12, fontweight='bold')

# Add legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_rain.get_legend_handles_labels()
lines3, labels3 = ax2_bikes.get_legend_handles_labels()
ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=9)

# Plot 3: Cloud cover, humidity, wind, and booked bikes
ax3 = axes[2]
ax3.plot(merged_df['timestamp'], merged_df['cloud_cover'], 
         label='Cloud Cover (%)', linewidth=2, color='#9C27B0', marker='o', markersize=4, alpha=0.8)
ax3.plot(merged_df['timestamp'], merged_df['relative_humidity_2m'], 
         label='Humidity (%)', linewidth=2, color='#00BCD4', marker='s', markersize=4, alpha=0.8)
ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')

ax3_wind = ax3.twinx()
ax3_wind.plot(merged_df['timestamp'], merged_df['wind_speed_10m'], 
              label='Wind Speed (m/s)', linewidth=2, color='#FF5722', marker='^', markersize=4, alpha=0.8)
ax3_wind.set_ylabel('Wind Speed (m/s)', fontsize=11, fontweight='bold', color='#FF5722')
ax3_wind.tick_params(axis='y', labelcolor='#FF5722')

# Add booked bikes on tertiary axis
ax3_bikes = ax3.twinx()
ax3_bikes.spines['right'].set_position(('outward', 60))
ax3_bikes.plot(merged_df['timestamp'], merged_df['booked_bikes'], 
               label='Booked Bikes', linewidth=2.5, color='#FF6B6B', marker='D', markersize=3, alpha=0.7, linestyle='--')
ax3_bikes.set_ylabel('Booked Bikes', fontsize=11, fontweight='bold', color='#FF6B6B')
ax3_bikes.tick_params(axis='y', labelcolor='#FF6B6B')

ax3.set_xlabel('Date & Time', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_title('Cloud Cover, Humidity, Wind Speed & Booked Bikes', fontsize=12, fontweight='bold')

# Add legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_wind.get_legend_handles_labels()
lines3, labels3 = ax3_bikes.get_legend_handles_labels()
ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=9)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Create a new figure with four subplots for period comparison
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig2.suptitle('Bike Rentals by Time Period - Rented Bikes, Temperature & Rain', fontsize=16, fontweight='bold')

# Define time periods based on analysis
# Period 1: Night (00:00-06:00) - Low activity (3.12 avg)
night_mask = (merged_df['timestamp'].dt.hour >= 0) & (merged_df['timestamp'].dt.hour < 6)
night_df = merged_df[night_mask].copy()

# Period 2: Morning (06:00-09:00) - Medium activity (17.46 avg)
morning_mask = (merged_df['timestamp'].dt.hour >= 6) & (merged_df['timestamp'].dt.hour < 9)
morning_df = merged_df[morning_mask].copy()

# Period 3: Peak Day (09:00-18:00) - Highest activity (30.95 avg)
peak_mask = (merged_df['timestamp'].dt.hour >= 9) & (merged_df['timestamp'].dt.hour < 18)
peak_df = merged_df[peak_mask].copy()

# Period 4: Evening (18:00-23:00) - Second peak (16.86 avg)
evening_mask = (merged_df['timestamp'].dt.hour >= 18) & (merged_df['timestamp'].dt.hour < 23)
evening_df = merged_df[evening_mask].copy()

periods = [
    (night_df, "Night: 00:00 - 06:00 (Low Activity: 3.12 bikes avg)", 0),
    (morning_df, "Morning: 06:00 - 09:00 (Medium Activity: 17.46 bikes avg)", 1),
    (peak_df, "Peak Day: 09:00 - 18:00 (Highest Activity: 30.95 bikes avg)", 2),
    (evening_df, "Evening: 18:00 - 23:00 (Second Peak: 16.86 bikes avg)", 3)
]

for period_data, title, idx in periods:
    ax_main = axes2[idx]
    
    # Plot booked bikes
    ax_main.plot(period_data['timestamp'], period_data['booked_bikes'], 
                 label='Booked Bikes', linewidth=2.5, color='#FF6B6B', marker='o', markersize=5, alpha=0.8)
    ax_main.fill_between(period_data['timestamp'], period_data['booked_bikes'], alpha=0.2, color='#FF6B6B')
    ax_main.set_ylabel('Booked Bikes', fontsize=11, fontweight='bold', color='#FF6B6B')
    ax_main.tick_params(axis='y', labelcolor='#FF6B6B')
    
    # Add temperature on secondary axis
    ax_temp = ax_main.twinx()
    ax_temp.plot(period_data['timestamp'], period_data['temperature_2m'], 
                 label='Temperature (°C)', linewidth=2.5, color='#FF9800', marker='s', markersize=4, alpha=0.8)
    ax_temp.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold', color='#FF9800')
    ax_temp.tick_params(axis='y', labelcolor='#FF9800')
    
    # Add rain on tertiary axis
    ax_rain = ax_main.twinx()
    ax_rain.spines['right'].set_position(('outward', 60))
    ax_rain.bar(period_data['timestamp'], period_data['rain'], 
                label='Rain (mm)', alpha=0.4, color='#2196F3', width=0.02)
    ax_rain.set_ylabel('Rain (mm)', fontsize=11, fontweight='bold', color='#2196F3')
    ax_rain.tick_params(axis='y', labelcolor='#2196F3')
    
    if idx == 3:
        ax_main.set_xlabel('Date & Time', fontsize=11, fontweight='bold')
    
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_title(title, fontsize=12, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    lines3, labels3 = ax_rain.get_legend_handles_labels()
    ax_main.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=9)

# Format x-axis for period figure
for ax in axes2:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('bike_weather_periods.png', dpi=300, bbox_inches='tight')
print("✓ Periods graph saved as 'bike_weather_periods.png'")

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('bike_weather_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved as 'bike_weather_analysis.png'")
print(f"\nData Summary:")
print(f"  Time range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
print(f"  Data points: {len(merged_df)}")
print(f"\nBooked Bikes Statistics:")
print(f"  Mean: {merged_df['booked_bikes'].mean():.2f}")
print(f"  Min: {merged_df['booked_bikes'].min():.2f}")
print(f"  Max: {merged_df['booked_bikes'].max():.2f}")
print(f"\nBooked Bikes Statistics by Period:")
print(f"\nNight (00:00-06:00) - Low Activity: 3.12 bikes avg")
print(f"  Mean: {night_df['booked_bikes'].mean():.2f}")
print(f"  Min: {night_df['booked_bikes'].min():.2f}")
print(f"  Max: {night_df['booked_bikes'].max():.2f}")
print(f"\nMorning (06:00-09:00) - Medium Activity: 17.46 bikes avg")
print(f"  Mean: {morning_df['booked_bikes'].mean():.2f}")
print(f"  Min: {morning_df['booked_bikes'].min():.2f}")
print(f"  Max: {morning_df['booked_bikes'].max():.2f}")
print(f"\nPeak Day (09:00-18:00) - Highest Activity: 30.95 bikes avg")
print(f"  Mean: {peak_df['booked_bikes'].mean():.2f}")
print(f"  Min: {peak_df['booked_bikes'].min():.2f}")
print(f"  Max: {peak_df['booked_bikes'].max():.2f}")
print(f"\nEvening (18:00-23:00) - Second Peak: 16.86 bikes avg")
print(f"  Mean: {evening_df['booked_bikes'].mean():.2f}")
print(f"  Min: {evening_df['booked_bikes'].min():.2f}")
print(f"  Max: {evening_df['booked_bikes'].max():.2f}")
print(f"\nWeather Statistics:")
print(f"  Avg Temperature: {merged_df['temperature_2m'].mean():.2f}°C")
print(f"  Avg Rain: {merged_df['rain'].mean():.2f}mm")
print(f"  Avg Humidity: {merged_df['relative_humidity_2m'].mean():.2f}%")

plt.show()
