import csv
import requests
import time
from datetime import datetime, timedelta
from collections import defaultdict

#convert the weather code
def get_weather_condition(weathercode):
    conditions = {
        0: 'sunny',
        1: 'sunny',
        2: 'partly_cloudy',
        3: 'cloudy',
        45: 'foggy',
        48: 'foggy',
        51: 'drizzle',
        53: 'drizzle',
        55: 'drizzle',
        61: 'rainy',
        63: 'rainy',
        65: 'rainy',
        66: 'freezing_rain',
        67: 'freezing_rain',
        71: 'snowy',
        73: 'snowy',
        75: 'snowy',
        77: 'snowy',
        80: 'rainy',
        81: 'rainy',
        82: 'rainy',
        85: 'snowy',
        86: 'snowy',
        95: 'thunderstorm',
        96: 'thunderstorm',
        99: 'thunderstorm'
    }
    return conditions.get(weathercode, 'unknown')

# convert date to week number
def get_week_number(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.isocalendar()[1]

# fetch the weather data
def fetch_weather_data(latitude, longitude, year=2024):
    """Fetch weather data from Open-Meteo API"""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': f'{year}-01-01',
        'end_date': f'{year}-12-31',
        'daily': 'temperature_2m_mean,weathercode',
        'timezone': 'America/Chicago'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


# get the information of the stadium
def process_stadium_weather():    
    stadiums = []
    with open('/Users/meganja/Desktop/cs238/cs238Project/data/stadium.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stadiums.append({
                'name': row['Name'],
                'latitude': float(row['Latitude']),
                'longitude': float(row['Longitude'])
            })
    
    # weekly aggregates
    temperature_data = []
    condition_data = []
    
    for stadium in stadiums:
        print(f"Processing {stadium['name']}...")
        
        weather_data = fetch_weather_data(stadium['latitude'], stadium['longitude'])
        
        if not weather_data or 'daily' not in weather_data:
            print(f"  Skipping {stadium['name']} - no data available")
            continue
        
        # Aggregate data by week
        weekly_temps = defaultdict(list)
        weekly_conditions = defaultdict(list)
        
        dates = weather_data['daily']['time']
        temps = weather_data['daily']['temperature_2m_mean']
        codes = weather_data['daily']['weathercode']
        
        for date, temp, code in zip(dates, temps, codes):
            week = get_week_number(date)
            if temp is not None:
                weekly_temps[week].append(temp)
            if code is not None:
                weekly_conditions[week].append(get_weather_condition(code))
        
        # Calculate weekly averages and most common conditions
        for week in range(1, 53):
            # Temperature average
            if week in weekly_temps:
                avg_temp = sum(weekly_temps[week]) / len(weekly_temps[week])
                temperature_data.append({
                    'stadium_name': stadium['name'],
                    'week_number': week,
                    'average_temperature': round(avg_temp, 2)
                })
            
            # Get mode of the condition
            if week in weekly_conditions:
                condition_counts = defaultdict(int)
                for condition in weekly_conditions[week]:
                    condition_counts[condition] += 1
                most_common = max(condition_counts.items(), key=lambda x: x[1])[0]
                condition_data.append({
                    'stadium_name': stadium['name'],
                    'week_number': week,
                    'condition': most_common
                })
        
        time.sleep(1)
    
    # write to temperature CSV
    with open('temperature.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['stadium_name', 'week_number', 'average_temperature'])
        writer.writeheader()
        writer.writerows(temperature_data)
    
    # write to conditions CSV
    with open('conditions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['stadium_name', 'week_number', 'condition'])
        writer.writeheader()
        writer.writerows(condition_data)
    
    print(f"\nDone!")

if __name__ == "__main__":
    process_stadium_weather()