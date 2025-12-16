# Vor dem Start müssen Bibliotheken installiert werden.
# Folgendes muss hierzu vor dem Start in das Terminal eingegeben werden:
# pip install openmeteo-requests
# pip install requests-cache retry-requests numpy pandas

import openmeteo_requests # Braucht man für den Abruf der Wetterdaten für Karlsruhe

import pandas as pd # Zur Datenanalyse
import requests_cache # Gut für API-Abfragen, damit Prozesse schneller Ablaufen
from retry_requests import retry # Bei Fehlern in der API-Abfrage

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 49.0094,
	"longitude": 8.4044,
	"start_date": "2025-09-14",
	"end_date": "2025-10-10",
	"hourly": ["temperature_2m", "rain", "snowfall", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "sunshine_duration", "is_day"],
	"timezone": "Europe/Berlin",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_rain = hourly.Variables(1).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
hourly_sunshine_duration = hourly.Variables(6).ValuesAsNumpy()
hourly_is_day = hourly.Variables(7).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m (°C)"] = hourly_temperature_2m
hourly_data["rain (mm)"] = hourly_rain
hourly_data["snowfall (mm)"] = hourly_snowfall
hourly_data["relative_humidity_2m (%)"] = hourly_relative_humidity_2m
hourly_data["cloud_cover (%)"] = hourly_cloud_cover
hourly_data["wind_speed_10m (km/h)"] = hourly_wind_speed_10m
hourly_data["sunshine_duration (seconds)"] = hourly_sunshine_duration
hourly_data["is_day (1 = yes, 0 = no)"] = hourly_is_day

hourly_dataframe = pd.DataFrame(data = hourly_data)

# Save data to CSV
hourly_dataframe.to_csv("weather_data.csv", index=False)
print("Daten gespeichert in: weather_data.csv")
