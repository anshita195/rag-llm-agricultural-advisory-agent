#!/usr/bin/env python3
"""
Reliable API fetcher using proven public APIs
- Weather: OpenWeatherMap (free tier)
- Soil: SoilGrids ISRIC (global soil data)
- Markets: Agmarknet CSV downloads
- Remote Sensing: NASA POWER API
- Policy: data.gov.in schemes
"""
import os
import requests
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from .datagovin_api_fetcher import DataGovInAPIFetcher
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ReliableAPIFetcher:
    def __init__(self, db_path: str = "data/agrisage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # API keys from environment
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        self.nasa_power_key = os.getenv('NASA_POWER_API_KEY')  # Optional
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgriSage/1.0 (Agricultural Advisory System)'
        })
        
        logger.info("🌱 Reliable API fetcher initialized")
    
    def fetch_openweather_data(self, locations: List[Dict]) -> List[Dict]:
        """Fetch weather from OpenWeatherMap API"""
        weather_data = []
        
        if not self.openweather_key:
            logger.info("OpenWeatherMap API key not configured, using NASA POWER weather data")
            return self._nasa_weather_fallback(locations)
        
        for location in locations:
            lat, lon = location['lat'], location['lon']
            district = location['district']
            
            # Current weather + 5-day forecast
            url = f"https://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_key,
                'units': 'metric',
                'cnt': 5  # 5 forecasts
            }
            
            try:
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for forecast in data['list']:
                        weather_data.append({
                            'district': district,
                            'date': datetime.fromtimestamp(forecast['dt']).strftime('%Y-%m-%d'),
                            'max_temp': forecast['main']['temp_max'],
                            'min_temp': forecast['main']['temp_min'],
                            'rainfall': forecast.get('rain', {}).get('3h', 0.0),
                            'humidity': forecast['main']['humidity'],
                            'wind_speed': forecast['wind']['speed'],
                            'precip_prob': forecast.get('pop', 0) * 100,
                            'description': forecast['weather'][0]['description'],
                            'source': 'OpenWeatherMap',
                            'url': f"https://openweathermap.org/city/{data['city']['id']}"
                        })
                    
                    logger.info(f"✅ OpenWeather data for {district}: {len(data['list'])} forecasts")
                    
                else:
                    logger.error(f"OpenWeather API error {response.status_code} for {district}")
                    
            except Exception as e:
                logger.error(f"OpenWeather fetch failed for {district}: {e}")
        
        return weather_data if weather_data else self._nasa_weather_fallback(locations)
    
    def fetch_soilgrids_data(self, locations: List[Dict]) -> List[Dict]:
        """Fetch soil data from SoilGrids ISRIC API"""
        soil_data = []
        
        for location in locations:
            lat, lon = location['lat'], location['lon']
            district = location['district']
            
            # SoilGrids REST API
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            params = {
                'lon': lon,
                'lat': lat,
                'property': ['phh2o', 'nitrogen', 'soc', 'sand', 'clay'],
                'depth': ['0-5cm', '5-15cm'],
                'value': 'mean'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract soil properties
                    properties = data.get('properties', {})
                    
                    soil_record = {
                        'district': district,
                        'lat': lat,
                        'lon': lon,
                        'pH': properties.get('phh2o', {}).get('0-5cm', {}).get('mean', 7.0) / 10,  # Convert from pH*10
                        'nitrogen': properties.get('nitrogen', {}).get('0-5cm', {}).get('mean', 1500) / 100,  # Convert cg/kg to %
                        'organic_carbon': properties.get('soc', {}).get('0-5cm', {}).get('mean', 15) / 10,  # Convert dg/kg to %
                        'sand_percent': properties.get('sand', {}).get('0-5cm', {}).get('mean', 30),
                        'clay_percent': properties.get('clay', {}).get('0-5cm', {}).get('mean', 25),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'SoilGrids_ISRIC',
                        'url': f"https://soilgrids.org/#!/?lat={lat}&lng={lon}&zoom=10"
                    }
                    
                    soil_data.append(soil_record)
                    logger.info(f"✅ SoilGrids data for {district}: pH {soil_record['pH']:.1f}")
                    
                else:
                    logger.error(f"SoilGrids API error {response.status_code} for {district}")
                    
            except Exception as e:
                logger.error(f"SoilGrids fetch failed for {district}: {e}")
        
        return soil_data if soil_data else self._soil_fallback(locations)
    
    def fetch_nasa_power_data(self, locations: List[Dict]) -> List[Dict]:
        """Fetch agricultural data from NASA POWER API"""
        agro_data = []
        
        for location in locations:
            lat, lon = location['lat'], location['lon']
            district = location['district']
            
            # NASA POWER Agroclimatology data
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                'parameters': 'T2M,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN',
                'community': 'AG',
                'longitude': lon,
                'latitude': lat,
                'start': (datetime.now() - timedelta(days=7)).strftime('%Y%m%d'),
                'end': datetime.now().strftime('%Y%m%d'),
                'format': 'JSON'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    parameters = data.get('properties', {}).get('parameter', {})
                    
                    # Get latest data
                    if parameters:
                        dates = list(parameters.get('T2M', {}).keys())[-3:]  # Last 3 days
                        
                        for date in dates:
                            agro_record = {
                                'district': district,
                                'date': f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                                'temperature': parameters.get('T2M', {}).get(date, 25.0),
                                'precipitation': parameters.get('PRECTOTCORR', {}).get(date, 0.0),
                                'humidity': parameters.get('RH2M', {}).get(date, 60.0),
                                'wind_speed': parameters.get('WS2M', {}).get(date, 5.0),
                                'solar_radiation': parameters.get('ALLSKY_SFC_SW_DWN', {}).get(date, 20.0),
                                'source': 'NASA_POWER',
                                'url': f"https://power.larc.nasa.gov/data-access-viewer/"
                            }
                            agro_data.append(agro_record)
                        
                        logger.info(f"✅ NASA POWER data for {district}: {len(dates)} days")
                    
                else:
                    logger.error(f"NASA POWER API error {response.status_code} for {district}")
                    
            except Exception as e:
                logger.error(f"NASA POWER fetch failed for {district}: {e}")
        
        return agro_data
    
    def fetch_agmarknet_csv(self) -> List[Dict]:
        """Download Agmarknet CSV data"""
        market_data = []
        
        # Use the working scraper
        try:
            print("\nFetching Market Data...")
            market_fetcher = DataGovInAPIFetcher()
            market_data = market_fetcher.fetch_market_prices_for_state('Uttarakhand')
            if market_data:
                market_fetcher.update_database(market_data)
                print(f"✅ Fetched and inserted {len(market_data)} market records.")
                return market_data
        except Exception as e:
            print(f"❌ Failed to fetch market data: {e}")
            return self.fallback_market_data()
        return [] # Return empty list if nothing is fetched

    def fallback_market_data(self):
        """No fallback - return empty if API fails."""
        logger.warning("Market API failed - no sample data fallback available")
        return []
        
        # Fallback to CSV URLs
        csv_urls = [
            "https://agmarknet.gov.in/Others/profile.aspx?ss=1&mi=3",  # Daily prices
            "https://agmarknet.gov.in/SearchCmmMkt.aspx"  # Market search
        ]
        
        for csv_url in csv_urls:
            try:
                response = self.session.get(csv_url, timeout=15)
                
                if response.status_code == 200 and 'csv' in response.headers.get('content-type', '').lower():
                    # Parse CSV directly
                    import io
                    df = pd.read_csv(io.StringIO(response.text))
                    
                    # Process market data
                    for _, row in df.head(50).iterrows():  # First 50 records
                        try:
                            market_record = {
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'commodity': str(row.iloc[1])[:30] if len(row) > 1 else 'Mixed',
                                'mandi': str(row.iloc[2])[:30] if len(row) > 2 else 'Unknown',
                                'district': str(row.iloc[3])[:20] if len(row) > 3 else 'Unknown',
                                'price': float(str(row.iloc[-1]).replace(',', '')) if str(row.iloc[-1]).replace(',', '').replace('.', '').isdigit() else 2500.0,
                                'source': 'Agmarknet_CSV',
                                'url': csv_url
                            }
                            market_data.append(market_record)
                            
                        except (ValueError, IndexError):
                            continue
                    
                    if market_data:
                        logger.info(f"✅ Agmarknet CSV: {len(market_data)} price records")
                        break
                        
            except Exception as e:
                logger.warning(f"Agmarknet CSV fetch failed for {csv_url}: {e}")
        
        return market_data if market_data else self._market_fallback()
    
    def update_database(self, weather_data: List[Dict], soil_data: List[Dict], 
                       agro_data: List[Dict], market_data: List[Dict]) -> bool:
        """Update database with fetched data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reliable_weather (
                    id INTEGER PRIMARY KEY,
                    district TEXT,
                    date TEXT,
                    max_temp REAL,
                    min_temp REAL,
                    rainfall REAL,
                    humidity REAL,
                    wind_speed REAL,
                    precip_prob REAL,
                    description TEXT,
                    source TEXT,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reliable_soil (
                    id INTEGER PRIMARY KEY,
                    district TEXT,
                    lat REAL,
                    lon REAL,
                    pH REAL,
                    nitrogen REAL,
                    organic_carbon REAL,
                    sand_percent REAL,
                    clay_percent REAL,
                    date TEXT,
                    source TEXT,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reliable_markets (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    commodity TEXT,
                    mandi TEXT,
                    district TEXT,
                    price REAL,
                    source TEXT,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear old data (keep last 30 days)
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            cursor.execute("DELETE FROM reliable_weather WHERE date < ?", (cutoff_date,))
            cursor.execute("DELETE FROM reliable_soil WHERE date < ?", (cutoff_date,))
            cursor.execute("DELETE FROM reliable_markets WHERE date < ?", (cutoff_date,))
            
            # Insert new data
            for record in weather_data:
                cursor.execute("""
                    INSERT INTO reliable_weather 
                    (district, date, max_temp, min_temp, rainfall, humidity, wind_speed, precip_prob, description, source, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (record['district'], record['date'], record['max_temp'], record['min_temp'],
                      record['rainfall'], record['humidity'], record['wind_speed'], record['precip_prob'],
                      record['description'], record['source'], record['url']))
            
            for record in soil_data:
                cursor.execute("""
                    INSERT INTO reliable_soil 
                    (district, lat, lon, pH, nitrogen, organic_carbon, sand_percent, clay_percent, date, source, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (record['district'], record['lat'], record['lon'], record['pH'], record['nitrogen'],
                      record['organic_carbon'], record['sand_percent'], record['clay_percent'],
                      record['date'], record['source'], record['url']))
            
            for record in market_data:
                cursor.execute("""
                    INSERT INTO reliable_markets 
                    (date, commodity, mandi, district, price, source, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (record['date'], record['commodity'], record['mandi'], record['district'],
                      record['price'], record['source'], record['url']))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Database updated: {len(weather_data)} weather, {len(soil_data)} soil, {len(market_data)} market records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database update failed: {e}")
            return False
    
    def _nasa_weather_fallback(self, locations: List[Dict]) -> List[Dict]:
        """Use NASA POWER data as weather fallback"""
        nasa_data = self.fetch_nasa_power_data(locations)
        weather_data = []
        
        for location in locations:
            district = location['district']
            # Find NASA data for this district
            district_nasa = [d for d in nasa_data if d['district'] == district]
            
            if district_nasa:
                latest = district_nasa[-1]  # Most recent
                weather_data.append({
                    'district': district,
                    'date': latest['date'],
                    'max_temp': latest['temperature'] + 3,  # Rough max temp
                    'min_temp': latest['temperature'] - 5,  # Rough min temp
                    'rainfall': latest['precipitation'],
                    'humidity': latest['humidity'],
                    'wind_speed': latest['wind_speed'],
                    'precip_prob': min(latest['precipitation'] * 20, 100),
                    'description': 'nasa power data',
                    'source': 'NASA_POWER_WEATHER',
                    'url': latest['url']
                })
            else:
                # No fallback - skip if no data available
                logger.warning(f"No weather data available for {district}")
                continue
        
        return weather_data
    
    def _soil_fallback(self, locations: List[Dict]) -> List[Dict]:
        """No fallback - return empty if API fails"""
        logger.warning("SoilGrids API failed - no fallback data available")
        return []
    
    def _market_fallback(self) -> List[Dict]:
        """No fallback - return empty if API fails"""
        logger.warning("Market API failed - no fallback data available")
        return []

def main():
    """Test the reliable API fetcher"""
    print("Testing Reliable API Fetcher...")
    
    # Test locations (Uttarakhand districts)
    locations = [
        {'district': 'Dehradun', 'lat': 30.3165, 'lon': 78.0322},
        {'district': 'Roorkee', 'lat': 29.8543, 'lon': 77.8880},
        {'district': 'Haridwar', 'lat': 29.9457, 'lon': 78.1642}
    ]
    
    fetcher = ReliableAPIFetcher()
    
    # Fetch all data
    weather_data = fetcher.fetch_openweather_data(locations)
    soil_data = fetcher.fetch_soilgrids_data(locations)
    agro_data = fetcher.fetch_nasa_power_data(locations)
    market_data = fetcher.fetch_agmarknet_csv()
    
    # Update database
    success = fetcher.update_database(weather_data, soil_data, agro_data, market_data)
    
    print(f"\nResults:")
    print(f"Weather records: {len(weather_data)}")
    print(f"Soil records: {len(soil_data)}")
    print(f"Agro records: {len(agro_data)}")
    print(f"Market records: {len(market_data)}")
    print(f"Database update: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
