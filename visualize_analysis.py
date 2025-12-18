import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# Load data
bikes_df = pd.read_csv('exports/city_summaries_hourly.csv')
weather_df = pd.read_csv('weather_data.csv')

# Convert timestamp columns to datetime
bikes_df['timestamp'] = pd.to_datetime(bikes_df['timestamp'])
weather_df['date'] = pd.to_datetime(weather_df['date'], utc=True)
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

# Group by timestamp and aggregate
merged_df = merged_df.groupby('timestamp').agg({
    'booked_bikes': 'mean',
    'total_bikes': 'mean',
    'available_bikes': 'mean',
    'temperature_2m': 'first',
    'rain': 'first',
    'wind_speed_10m': 'first',
    'relative_humidity_2m': 'first'
}).reset_index()

merged_df = merged_df.sort_values('timestamp')

# Add day type classification
def get_day_type(date):
    """Classify as Workday (Mo-Fr), Weekend (Sa-Su), or Holiday"""
    day_name = date.strftime('%A')
    weekday = date.weekday()  # 0=Monday, 6=Sunday
    
    # German holidays in 2025 (relevant to data range Sep-Oct)
    holidays = [
        pd.Timestamp('2025-09-15'),  # Placeholder, add actual holidays if needed
        pd.Timestamp('2025-10-03'),  # German Unity Day
    ]
    
    if date in holidays:
        return 'Holiday'
    elif weekday >= 5:  # Saturday or Sunday
        return 'Weekend'
    else:
        return 'Workday'

merged_df['day_type'] = merged_df['timestamp'].apply(get_day_type)

# ============================================================================
# FIGURE 1: Time Series by Day Type
# ============================================================================
fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig1.suptitle('Bike Rentals vs Weather by Day Type', fontsize=16, fontweight='bold')

day_types = ['Workday', 'Weekend', 'Holiday']
colors = {'Workday': '#1f77b4', 'Weekend': '#ff7f0e', 'Holiday': '#d62728'}

for idx, day_type in enumerate(day_types):
    data = merged_df[merged_df['day_type'] == day_type]
    ax = axes1[idx]
    
    if len(data) == 0:
        ax.text(0.5, 0.5, f'No {day_type} data', ha='center', va='center', transform=ax.transAxes)
        continue
    
    # Plot booked bikes
    ax.plot(data['timestamp'], data['booked_bikes'], 
            label='Booked Bikes', linewidth=2.5, color=colors[day_type], marker='o', markersize=3, alpha=0.8)
    ax.fill_between(data['timestamp'], data['booked_bikes'], alpha=0.2, color=colors[day_type])
    ax.set_ylabel('Booked Bikes', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(f'{day_type}', fontsize=12, fontweight='bold')
    
    # Add temperature overlay on secondary axis
    ax_temp = ax.twinx()
    ax_temp.plot(data['timestamp'], data['temperature_2m'], 
                 label='Temperature (°C)', linewidth=2, color='#FF9800', linestyle='--', alpha=0.7)
    ax_temp.set_ylabel('Temperature (°C)', fontsize=10, color='#FF9800')
    ax_temp.tick_params(axis='y', labelcolor='#FF9800')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    if idx == 2:
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('01_day_type_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_day_type_comparison.png")

# ============================================================================
# FIGURE 2: Temperature Impact Analysis
# ============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Impact of Temperature on Bike Rentals', fontsize=14, fontweight='bold')

# Define temperature ranges
merged_df['temp_range'] = pd.cut(merged_df['temperature_2m'], 
                                 bins=[-np.inf, 10, 15, 20, np.inf],
                                 labels=['<10°C', '10-15°C', '15-20°C', '>20°C'])

# Plot 1: Average booked bikes by temperature range and day type
ax = axes2[0]
temp_day_stats = merged_df.groupby(['temp_range', 'day_type'])['booked_bikes'].mean().unstack()
temp_day_stats.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#d62728'], width=0.8)
ax.set_title('Avg Booked Bikes by Temperature Range', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Booked Bikes', fontsize=10)
ax.set_xlabel('Temperature Range', fontsize=10)
ax.legend(title='Day Type', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Scatter plot - Temperature vs Booked Bikes (colored by day type)
ax = axes2[1]
for day_type, color in colors.items():
    data = merged_df[merged_df['day_type'] == day_type]
    ax.scatter(data['temperature_2m'], data['booked_bikes'], 
              label=day_type, alpha=0.5, s=30, color=color)
ax.set_title('Temperature vs Booked Bikes', fontsize=11, fontweight='bold')
ax.set_xlabel('Temperature (°C)', fontsize=10)
ax.set_ylabel('Booked Bikes', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Box plot - Distribution by temperature range and day type
ax = axes2[2]
data_for_box = []
labels_for_box = []
for temp_range in ['<10°C', '10-15°C', '15-20°C', '>20°C']:
    for day_type in ['Workday', 'Weekend', 'Holiday']:
        subset = merged_df[(merged_df['temp_range'] == temp_range) & (merged_df['day_type'] == day_type)]['booked_bikes']
        if len(subset) > 0:
            data_for_box.append(subset)
            labels_for_box.append(f'{temp_range}\n{day_type}')

bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
for patch, label in zip(bp['boxes'], labels_for_box):
    if 'Workday' in label:
        patch.set_facecolor('#1f77b4')
    elif 'Weekend' in label:
        patch.set_facecolor('#ff7f0e')
    else:
        patch.set_facecolor('#d62728')
ax.set_title('Distribution by Temperature & Day Type', fontsize=11, fontweight='bold')
ax.set_ylabel('Booked Bikes', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('02_temperature_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_temperature_impact.png")

# ============================================================================
# FIGURE 3: Precipitation Impact Analysis
# ============================================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Impact of Precipitation on Bike Rentals', fontsize=14, fontweight='bold')

# Define rain conditions
merged_df['rain_condition'] = merged_df['rain'].apply(lambda x: 'Rainy' if x > 0.5 else 'Dry')

# Plot 1: Average booked bikes by rain condition and day type
ax = axes3[0]
rain_day_stats = merged_df.groupby(['rain_condition', 'day_type'])['booked_bikes'].mean().unstack()
rain_day_stats.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#d62728'], width=0.6)
ax.set_title('Avg Booked Bikes: Rainy vs Dry', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Booked Bikes', fontsize=10)
ax.set_xlabel('Weather Condition', fontsize=10)
ax.legend(title='Day Type', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# Plot 2: Scatter plot - Rain vs Booked Bikes
ax = axes3[1]
for day_type, color in colors.items():
    data = merged_df[merged_df['day_type'] == day_type]
    ax.scatter(data['rain'], data['booked_bikes'], 
              label=day_type, alpha=0.5, s=30, color=color)
ax.set_title('Precipitation vs Booked Bikes', fontsize=11, fontweight='bold')
ax.set_xlabel('Rain (mm)', fontsize=10)
ax.set_ylabel('Booked Bikes', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_precipitation_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_precipitation_impact.png")

# ============================================================================
# FIGURE 4: Correlation Analysis
# ============================================================================
fig4 = plt.figure(figsize=(14, 8))
gs = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
fig4.suptitle('Quantitative Weather Impact Analysis', fontsize=14, fontweight='bold')

# Calculate correlations for each day type
correlations = {}
for day_type in day_types:
    data = merged_df[merged_df['day_type'] == day_type]
    if len(data) > 1:
        corr_temp = pearsonr(data['temperature_2m'].dropna(), data.loc[data['temperature_2m'].notna(), 'booked_bikes'])[0]
        corr_rain = pearsonr(data['rain'].dropna(), data.loc[data['rain'].notna(), 'booked_bikes'])[0]
        corr_wind = pearsonr(data['wind_speed_10m'].dropna(), data.loc[data['wind_speed_10m'].notna(), 'booked_bikes'])[0]
        corr_humidity = pearsonr(data['relative_humidity_2m'].dropna(), data.loc[data['relative_humidity_2m'].notna(), 'booked_bikes'])[0]
        
        correlations[day_type] = {
            'Temperature': corr_temp,
            'Rain': corr_rain,
            'Wind': corr_wind,
            'Humidity': corr_humidity
        }

# Plot 1: Correlation heatmap
ax = fig4.add_subplot(gs[0, :])
corr_df = pd.DataFrame(correlations).T
im = ax.imshow(corr_df.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_df.columns)))
ax.set_yticks(np.arange(len(corr_df.index)))
ax.set_xticklabels(corr_df.columns)
ax.set_yticklabels(corr_df.index)
ax.set_title('Correlation Coefficients: Weather Factors vs Booked Bikes', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(corr_df.index)):
    for j in range(len(corr_df.columns)):
        text = ax.text(j, i, f'{corr_df.values[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax, label='Correlation Coefficient')

# Plot 2: Bar chart of correlations
ax = fig4.add_subplot(gs[1, :])
x = np.arange(len(corr_df.columns))
width = 0.25

for i, day_type in enumerate(day_types):
    values = [correlations[day_type][col] for col in corr_df.columns]
    ax.bar(x + i*width, values, width, label=day_type, color=list(colors.values())[i], alpha=0.8)

ax.set_ylabel('Correlation Coefficient', fontsize=10)
ax.set_xlabel('Weather Factor', fontsize=10)
ax.set_title('Correlation Strength by Day Type', fontsize=11, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(corr_df.columns)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('04_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_correlation_analysis.png")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

for day_type in day_types:
    data = merged_df[merged_df['day_type'] == day_type]
    print(f"\n{day_type.upper()}:")
    print(f"  Data points: {len(data)}")
    print(f"  Booked Bikes - Mean: {data['booked_bikes'].mean():.2f}, "
          f"Min: {data['booked_bikes'].min():.2f}, Max: {data['booked_bikes'].max():.2f}")
    print(f"  Temperature - Mean: {data['temperature_2m'].mean():.2f}°C, "
          f"Range: {data['temperature_2m'].min():.2f}°C to {data['temperature_2m'].max():.2f}°C")
    print(f"  Rainy hours: {(data['rain'] > 0.5).sum()} ({(data['rain'] > 0.5).sum()/len(data)*100:.1f}%)")

print("\n" + "="*70)
print("CORRELATION COEFFICIENTS")
print("="*70)
for day_type in day_types:
    print(f"\n{day_type.upper()}:")
    for factor, corr in correlations[day_type].items():
        strength = "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"  {factor:12} → {corr:7.3f} ({direction:8} - {strength})")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Find strongest correlations
all_corrs = []
for day_type, factors in correlations.items():
    for factor, corr in factors.items():
        all_corrs.append((day_type, factor, corr))

all_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nStrongest Correlations (by absolute value):")
for day_type, factor, corr in all_corrs[:5]:
    print(f"  {factor:12} on {day_type:10} → {corr:7.3f}")

# Temperature impact
print("\nTemperature Impact:")
for temp_range in ['<10°C', '10-15°C', '15-20°C', '>20°C']:
    avg = merged_df[merged_df['temp_range'] == temp_range]['booked_bikes'].mean()
    count = len(merged_df[merged_df['temp_range'] == temp_range])
    print(f"  {temp_range:10} → {avg:6.2f} bikes avg ({count} observations)")

# Rain impact
print("\nRain Impact:")
dry_avg = merged_df[merged_df['rain_condition'] == 'Dry']['booked_bikes'].mean()
rainy_avg = merged_df[merged_df['rain_condition'] == 'Rainy']['booked_bikes'].mean()
print(f"  Dry conditions  → {dry_avg:6.2f} bikes avg")
print(f"  Rainy conditions → {rainy_avg:6.2f} bikes avg")
print(f"  Difference      → {dry_avg - rainy_avg:6.2f} bikes ({(dry_avg - rainy_avg)/rainy_avg*100:.1f}% more)")

print("\n✓ Analysis complete!")
