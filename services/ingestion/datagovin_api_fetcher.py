#!/usr/bin/env python3
"""
Data.gov.in API fetcher for reliable mandi prices
Uses official JSON endpoints instead of HTML scraping
"""
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DataGovInAPIFetcher:
    def __init__(self, db_path: str = "data/agrisage.db"):
        self.db_path = Path(db_path)
        self.api_key = os.getenv('DATA_GOV_IN_API_KEY')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgriSage/1.0 (Agricultural Advisory System)',
            'Accept': 'application/json'
        })
        
        # Data.gov.in endpoints for agricultural data
        self.endpoints = {
            'mandi_prices': '9ef84268-d588-465a-a308-a864a43d0070',  # Daily mandi prices
            'crop_production': '99b3d0b1-6f8e-4b8e-9c8a-1234567890ab',  # Crop production data
            'rainfall': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'  # Rainfall data
        }
        
        # Target states and districts with fallback strategy
        self.target_states = ['Uttarakhand', 'Uttar Pradesh', 'Punjab', 'Haryana']
        self.target_districts = ['Dehradun', 'Haridwar', 'Roorkee', 'Rishikesh', 'Pauri Garhwal']
        
        # Region-aware fallback mapping: state -> [immediate neighbors, nearby states, wider region]
        self.regional_fallback = {
            'uttarakhand': {
                'immediate': ['uttar pradesh', 'himachal pradesh'],
                'nearby': ['haryana', 'punjab', 'delhi'],
                'regional': ['rajasthan', 'bihar', 'jharkhand'],
                'distant': ['madhya pradesh', 'chhattisgarh', 'west bengal'],
                'national': ['gujarat', 'maharashtra', 'karnataka', 'andhra pradesh']
            },
            'uttar pradesh': {
                'immediate': ['uttarakhand', 'bihar', 'madhya pradesh'],
                'nearby': ['haryana', 'delhi', 'rajasthan'],
                'regional': ['punjab', 'jharkhand', 'chhattisgarh'],
                'distant': ['west bengal', 'himachal pradesh'],
                'national': ['gujarat', 'maharashtra', 'karnataka']
            },
            'haryana': {
                'immediate': ['punjab', 'delhi', 'uttar pradesh'],
                'nearby': ['uttarakhand', 'rajasthan', 'himachal pradesh'],
                'regional': ['madhya pradesh', 'bihar'],
                'distant': ['gujarat', 'jharkhand'],
                'national': ['maharashtra', 'karnataka', 'west bengal']
            },
            'punjab': {
                'immediate': ['haryana', 'himachal pradesh'],
                'nearby': ['delhi', 'uttarakhand', 'rajasthan'],
                'regional': ['uttar pradesh', 'jammu and kashmir'],
                'distant': ['madhya pradesh', 'gujarat'],
                'national': ['maharashtra', 'karnataka', 'bihar']
            }
        }
        
        # Commodity name mapping for query normalization (with Hindi synonyms)
        self.commodity_map = {
            'rice': ['Rice', 'Paddy(Dhan)', 'Paddy', 'Basmati', 'Non-Basmati', 'धान', 'चावल'],
            'wheat': ['Wheat', 'Gehun', 'Wheat Flour', 'गेहूं', 'गेहुँ'],
            'mustard': ['Mustard', 'Sarson', 'Mustard Seed', 'Rape Seed', 'सरसों'],
            'maize': ['Maize', 'Corn', 'Makka', 'मक्का'],
            'sugarcane': ['Sugarcane', 'Sugar Cane', 'गन्ना'],
            'soybean': ['Soybean', 'Soya Bean', 'सोयाबीन'],
            'cotton': ['Cotton', 'Kapas', 'कपास'],
            'onion': ['Onion', 'Pyaz', 'प्याज'],
            'potato': ['Potato', 'Aloo', 'आलू'],
            'tomato': ['Tomato', 'Tamatar', 'टमाटर']
        }
        
        # Location mapping for district/mandi normalization
        self.location_map = {
            'roorkee': ['Roorkee', 'Haridwar', 'Hardwar'],
            'dehradun': ['Dehradun', 'Dehra Dun'],
            'rishikesh': ['Rishikesh', 'Haridwar'],
            'pauri': ['Pauri Garhwal', 'Pauri'],
            'uttarakhand': ['Uttarakhand', 'UK']
        }
    
    def fetch_market_prices_for_state(self, primary_state: str, limit: int = 2000) -> List[Dict]:
        """Fetch mandi prices, trying a primary state and then falling back to others."""
        if not self.api_key:
            logger.error("❌ DATA_GOV_IN_API_KEY not found in environment. Cannot fetch market data.")
            return []

        fallback_config = self.regional_fallback.get(primary_state.lower(), self.regional_fallback['uttarakhand'])
        states_to_try = (
            [primary_state] + 
            fallback_config.get('immediate', []) + 
            fallback_config.get('nearby', [])
        )

        all_records = []
        for state in states_to_try:
            logger.info(f"🌐 Attempting to fetch market prices for {state}...")
            try:
                url = f"https://api.data.gov.in/resource/{self.endpoints['mandi_prices']}"
                params = {
                    'api-key': self.api_key,
                    'format': 'json',
                    'offset': 0,
                    'limit': limit,
                    f'filters[state]': state
                }
                
                response = self.session.get(url, params=params, timeout=45)
                response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)

                data = response.json()
                if 'records' in data and data['records']:
                    records = data['records']
                    logger.info(f"✅ Retrieved {len(records)} raw records for {state}.")
                    
                    processed_for_state = []
                    for record in records:
                        processed_record = self._process_mandi_record(record)
                        if processed_record:
                            processed_for_state.append(processed_record)
                    
                    if processed_for_state:
                        logger.info(f"🎯 Processed {len(processed_for_state)} valid records for {state}. Stopping search.")
                        all_records.extend(processed_for_state)
                        return all_records # Success, so we stop and return the data
                else:
                    logger.warning(f"⚠️ No market data records found for {state}. Trying next state.")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    logger.error("❌ API key is invalid or expired. Aborting market data fetch.")
                    return [] # Stop trying if key is bad
                logger.error(f"❌ HTTP Error fetching data for {state}: {e}. Trying next state.")
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Request failed for {state}: {e}. Trying next state.")
            except Exception as e:
                logger.error(f"❌ An unexpected error occurred for {state}: {e}. Trying next state.")
        
        if not all_records:
            logger.error(f"❌ All attempts to fetch market data failed for primary and fallback states.")
        
        return all_records
    
    def _process_mandi_record(self, raw_record: Dict) -> Optional[Dict]:
        """Process raw API record into standardized format"""
        try:
            # Map API fields to our schema
            # Note: Field names may vary - adjust based on actual API response
            record = {
                'date': self._parse_date(raw_record.get('arrival_date', raw_record.get('date', ''))),
                'state': raw_record.get('state', '').strip(),
                'district': raw_record.get('district', '').strip(),
                'mandi': raw_record.get('market', raw_record.get('mandi_name', '')).strip(),
                'commodity': raw_record.get('commodity', '').strip(),
                'variety': raw_record.get('variety', 'Common').strip(),
                'grade': raw_record.get('grade', 'FAQ').strip(),
                'min_price': self._parse_price(raw_record.get('min_price', raw_record.get('minimum', 0))),
                'max_price': self._parse_price(raw_record.get('max_price', raw_record.get('maximum', 0))),
                'modal_price': self._parse_price(raw_record.get('modal_price', raw_record.get('mode', 0))),
                'arrival': raw_record.get('arrival_tonnes', raw_record.get('arrival', '0')),
                'source': 'DataGovIn_API',
                'url': 'https://data.gov.in'
            }
            
            # Use modal price as primary, fallback to max, then min
            record['price'] = record['modal_price'] or record['max_price'] or record['min_price']
            
            # Validate essential fields
            if not record['commodity'] or not record['state'] or record['price'] <= 0:
                return None
            
            return record
            
        except Exception as e:
            logger.debug(f"Failed to process record: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> str:
        """Parse various date formats to YYYY-MM-DD"""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Try common formats
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S']:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If all formats fail, return today
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')
    
    def _parse_price(self, price_value) -> float:
        """Parse price value to float"""
        try:
            if isinstance(price_value, (int, float)):
                return float(price_value)
            
            if isinstance(price_value, str):
                # Remove currency symbols and commas
                cleaned = price_value.replace('₹', '').replace('Rs.', '').replace(',', '').strip()
                if cleaned and cleaned.upper() not in ['NR', 'NA', '-']:
                    return float(cleaned)
            
            return 0.0
            
        except (ValueError, TypeError):
            return 0.0
    
    def _is_relevant_record(self, record: Dict) -> bool:
        """Check if record is relevant (keep all staples + target regions)"""
        state = record.get('state', '').lower()
        district = record.get('district', '').lower()
        commodity = record.get('commodity', '').lower()
        
        # Always keep important staples from any state
        important_commodities = ['rice', 'wheat', 'paddy', 'mustard', 'maize', 'sugarcane', 'cotton']
        if any(c in commodity for c in important_commodities):
            return True
        
        # Keep target states/districts for all commodities
        if any(target_state.lower() in state for target_state in self.target_states):
            return True
        
        if any(target_district.lower() in district for target_district in self.target_districts):
            return True
        
        return False
    
    def update_database(self, market_data: List[Dict]) -> bool:
        """Update real_mandi_prices table with API data"""
        if not market_data:
            logger.warning("No market data to update")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS real_mandi_prices (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    commodity TEXT,
                    mandi TEXT,
                    district TEXT,
                    state TEXT,
                    variety TEXT,
                    grade TEXT,
                    min_price REAL,
                    max_price REAL,
                    modal_price REAL,
                    price REAL,
                    arrival TEXT,
                    source TEXT,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing API data (keep scraped data as fallback)
            cursor.execute("DELETE FROM real_mandi_prices WHERE source = 'DataGovIn_API'")
            
            # Insert new records
            for record in market_data:
                cursor.execute("""
                    INSERT INTO real_mandi_prices 
                    (date, commodity, mandi, district, state, variety, grade, 
                     min_price, max_price, modal_price, price, arrival, source, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record['date'], record['commodity'], record['mandi'], 
                    record['district'], record['state'], record['variety'], record['grade'],
                    record['min_price'], record['max_price'], record['modal_price'], 
                    record['price'], record['arrival'], record['source'], record['url']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Updated real_mandi_prices: {len(market_data)} API records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database update failed: {e}")
            return False
    
    def get_price_for_query(self, commodity: str, location: str = None) -> Optional[Dict]:
        """Get specific price for farmer query with smart matching"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Normalize commodity using mapping
            commodity_variants = self._get_commodity_variants(commodity)
            location_variants = self._get_location_variants(location) if location else []
            
            # Try exact matches first
            result = self._query_with_variants(cursor, commodity_variants, location_variants, exact=True)
            
            # If no exact match, try fuzzy matching
            if not result:
                result = self._query_with_variants(cursor, commodity_variants, location_variants, exact=False)
            
            # If still no match and location specified, try neighboring states
            if not result and location:
                result = self._query_with_fallback_states(cursor, commodity_variants)
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get price for query: {e}")
            return None
    
    def format_farmer_response(self, commodity: str, location: str, result: Optional[Dict]) -> str:
        """Format farmer-friendly price response"""
        if not result:
            return f"❌ No price data available for {commodity.title()}{f' in {location.title()}' if location else ''} today. Try a broader search or check again later."
        
        # Format price with currency and consistent units
        price_display = f"₹{result['price']:.0f} per quintal"
        
        # Format date with "as of" prefix
        try:
            from datetime import datetime
            date_obj = datetime.strptime(result['date'], '%Y-%m-%d')
            date_display = f"as of {date_obj.strftime('%d %b %Y')}"
        except:
            date_display = f"as of {result['date']}"
        
        # Handle different match types
        match_type = result.get('match_type', 'exact')
        
        if match_type == 'exact':
            return f"✅ {result['commodity']} price in {result['district']}: {price_display} at {result['mandi']} ({date_display})"
        
        elif 'fallback' in match_type:
            tier = result.get('tier', 'neighboring')
            
            # Create farmer-friendly fallback message
            if location:
                response = f"📍 No {result['commodity']} price found in {location.title()} today.\n"
            else:
                response = f"📍 Local {result['commodity']} price not available today.\n"
            
            # Add available price info
            response += f"✅ Nearest available: {result['commodity']} at {result['mandi']}, {result['district']} ({result['state']}) — {price_display} ({date_display})\n"
            
            # Add context based on tier with Hindi commodity names
            commodity_hindi = self._get_hindi_name(result['commodity'])
            hindi_suffix = f" ({commodity_hindi})" if commodity_hindi else ""
            
            if tier == 'immediate':
                response += f"ℹ️ Using neighboring {result['state']} data (shares border with Uttarakhand){hindi_suffix}"
            elif tier == 'nearby':
                response += f"ℹ️ Using nearby {result['state']} data (similar climate zone){hindi_suffix}"
            elif tier == 'regional':
                response += f"ℹ️ Using regional {result['state']} data for comparison{hindi_suffix}"
            elif tier == 'distant':
                response += f"ℹ️ Using distant {result['state']} data (limited local availability){hindi_suffix}"
            elif tier == 'national':
                response += f"ℹ️ Using national benchmark from {result['state']}{hindi_suffix}"
            else:
                response += f"ℹ️ Using available data from {result['state']}{hindi_suffix}"
            
            return response
        
        else:  # fuzzy match
            return f"✅ {result['commodity']} ({result['variety']}) in {result['district']}: {price_display} at {result['mandi']} ({date_display})"
    
    def _query_with_fallback_states(self, cursor, commodity_variants: List[str]) -> Optional[Dict]:
        """Query with region-aware cascading fallback based on geographic proximity"""
        # Determine primary state context (default to Uttarakhand for AgriSage)
        primary_state = 'uttarakhand'
        
        # Get region-aware fallback tiers
        fallback_config = self.regional_fallback.get(primary_state, self.regional_fallback['uttarakhand'])
        
        # Build cascading fallback tiers with geographic logic
        fallback_tiers = [
            (fallback_config['immediate'], 'immediate'),
            (fallback_config['nearby'], 'nearby'), 
            (fallback_config['regional'], 'regional'),
            (fallback_config['distant'], 'distant'),
            (fallback_config['national'], 'national'),
            ([], 'any_state')  # Final fallback to any available state
        ]
        
        for states, tier_name in fallback_tiers:
            commodity_conditions = []
            params = []
            
            for variant in commodity_variants:
                commodity_conditions.append("LOWER(commodity) LIKE ?")
                params.append(f"%{variant.lower()}%")
            
            query = f"""
                SELECT commodity, district, mandi, price, variety, date, source, state
                FROM real_mandi_prices 
                WHERE ({' OR '.join(commodity_conditions)})
            """
            
            # Add state filter only for specific tiers
            if states:
                state_conditions = []
                for state in states:
                    state_conditions.append("LOWER(state) LIKE ?")
                    params.append(f"%{state.lower()}%")
                query += f" AND ({' OR '.join(state_conditions)})"
            
            query += " ORDER BY date DESC, price DESC LIMIT 3"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            if results:
                commodity, district, mandi, price, variety, date, source, state = results[0]
                
                note_map = {
                    'immediate': f'No local data found, showing price from neighboring {state}',
                    'nearby': f'No local data found, showing price from nearby {state}', 
                    'regional': f'No local data found, showing price from regional market {state}',
                    'distant': f'No local data found, showing price from distant market {state}',
                    'national': f'Using national benchmark from {state}',
                    'any_state': f'Using available data from {state}'
                }
                
                return {
                    'commodity': commodity,
                    'district': district,
                    'mandi': mandi,
                    'price': price,
                    'variety': variety,
                    'date': date,
                    'source': source,
                    'state': state,
                    'alternatives': len(results) - 1,
                    'match_type': f'{tier_name}_fallback',
                    'note': note_map[tier_name],
                    'tier': tier_name
                }
        
        return None

def main():
    """Test the data.gov.in API fetcher with farmer-friendly responses"""
    print("Testing Data.gov.in API Fetcher with Farmer-Friendly Responses...")
    
    fetcher = DataGovInAPIFetcher()
    
    if not fetcher.api_key:
        print("DATA_GOV_IN_API_KEY not found in .env file")
        print("Please register at https://data.gov.in and add your API key to .env")
        return
    
    # Fetch mandi prices
    market_data = fetcher.fetch_market_prices_for_state('Uttarakhand', limit=500)
    
    if market_data:
        print(f"\nFetched {len(market_data)} relevant mandi records:")
        for record in market_data[:5]:
            print(f"  {record['commodity']} at {record['mandi']}, {record['district']} ({record['state']}): ₹{record['price']}")
        
        success = fetcher.update_database(market_data)
        print(f"Database update: {'Success' if success else 'Failed'}")
        
        # Show data coverage summary
        print(f"\nData Coverage Summary:")
        conn = sqlite3.connect(fetcher.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT commodity, state, COUNT(*) as count
            FROM real_mandi_prices 
            WHERE source = 'DataGovIn_API'
            GROUP BY commodity, state
            ORDER BY count DESC
            LIMIT 10
        """)
        coverage = cursor.fetchall()
        for commodity, state, count in coverage:
            print(f"  {commodity} in {state}: {count} records")
        conn.close()
        
        # Test enhanced query functionality with region-aware fallback
        print(f"\nTesting region-aware price queries:")
        test_queries = [
            ('rice', 'roorkee'),
            ('wheat', 'dehradun'), 
            ('mustard', 'uttarakhand'),
            ('paddy', 'haridwar'),  # Test commodity mapping
            ('sarson', None),       # Test without location
            ('maize', 'uk'),        # Test location mapping
            ('cotton', 'uttarakhand')  # Test distant fallback
        ]
        
        for commodity, location in test_queries:
            result = fetcher.get_price_for_query(commodity, location)
            farmer_response = fetcher.format_farmer_response(commodity, location, result)
            print(f"\nQuery: {commodity}{f' in {location}' if location else ''}")
            print(f"Response: {farmer_response}")
    else:
        print("No market data fetched from API")
        print("Troubleshooting:")
        print("  1. Check DATA_GOV_IN_API_KEY in .env")
        print("  2. Verify internet connection")
        print("  3. Try again later (API may be temporarily down)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
