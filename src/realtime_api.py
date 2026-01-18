import requests
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import OPENWEATHERMAP_API_KEY
except ImportError:
    OPENWEATHERMAP_API_KEY = None

class WeatherAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or OPENWEATHERMAP_API_KEY or os.getenv('OPENWEATHER_API_KEY', 'demo')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, lat, lon):
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            resp = requests.get(self.base_url, params=params, timeout=5) 
            if resp.status_code == 200:
                data = resp.json()
                w_main = data['weather'][0]['main'].lower()
                w_map = {
                    'clear': 'Clear', 'clouds': 'Cloudy', 'rain': 'Rain',
                    'drizzle': 'Rain', 'thunderstorm': 'Rain', 'snow': 'Snow',
                    'mist': 'Fog', 'fog': 'Fog', 'haze': 'Fog'
                }               
                return {
                    'condition': w_map.get(w_main, 'Clear'),
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'description': data['weather'][0]['description'],
                    'source': 'API'
                }
            else:
                return self._fallback_weather()         
        except Exception as e:
            return self._fallback_weather()
    
    def _fallback_weather(self):
        import random
        conditions = ['Clear', 'Cloudy', 'Rain', 'Fog']
        weights = [0.5, 0.3, 0.15, 0.05]
        return {
            'condition': random.choices(conditions, weights)[0],
            'temperature': random.uniform(20, 35),
            'humidity': random.randint(40, 90),
            'wind_speed': random.uniform(0, 15),
            'description': 'simulated data',
            'source': 'Fallback (API unavailable)'
        }


class TrafficAPI:    
    def __init__(self):
        self.peak_hours = [(7, 10), (17, 20)]

    def get_traffic_level(self, lat, lon, current_time=None):
        if current_time is None:
            current_time = datetime.now()
        hour = current_time.hour
        day = current_time.weekday()
        base_level = 4
        for start, end in self.peak_hours:
            if start <= hour <= end:
                base_level += 3
                break
        if day >= 5:
            base_level -= 2
        metro_cities = {
            (19.07, 72.87): 2,   # Mumbai
            (28.61, 77.20): 2,   # Delhi
            (12.97, 77.59): 2,   # Bangalore
            (22.57, 88.36): 1,   # Kolkata
            (13.08, 80.27): 1,   # Chennai
        }
        
        for (city_lat, city_lon), bonus in metro_cities.items():
            if abs(lat - city_lat) < 0.5 and abs(lon - city_lon) < 0.5:
                base_level += bonus
                break
        traffic_level = max(1, min(10, base_level))
        return {
            'level': traffic_level,
            'description': self._get_description(traffic_level),
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Traffic Estimation Model'
        }
    
    def _get_description(self, level):
        if level <= 3:
            return 'Light traffic'
        elif level <= 5:
            return 'Moderate traffic'
        elif level <= 7:
            return 'Heavy traffic'
        else:
            return 'Severe congestion'

class RealtimeDataService:    
    def __init__(self, weather_api_key=None):
        self.weather_api = WeatherAPI(weather_api_key)
        self.traffic_api = TrafficAPI()
    
    def get_conditions(self, lat, lon):
        weather = self.weather_api.get_weather(lat, lon)
        traffic = self.traffic_api.get_traffic_level(lat, lon)
        return {
            'weather': weather,
            'traffic': traffic,
            'coordinates': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

if __name__ == "__main__":
    service = RealtimeDataService()  
    result = service.get_conditions(19.0760, 72.8777) 
    print("Real-time Conditions for Mumbai:")
    print(f"Weather: {result['weather']['condition']} ({result['weather']['description']})")
    print(f"Temperature: {result['weather']['temperature']}°C")
    print(f"Traffic Level: {result['traffic']['level']}/10 ({result['traffic']['description']})")
    print(f"Source: {result['weather']['source']}")
