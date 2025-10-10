#!/usr/bin/env python3
"""
Build Chroma vector index for AgriSage RAG system
"""
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import pandas as pd

def load_data_from_db():
    """Load data from SQLite database for indexing"""
    db_path = Path("data/agrisage.db")
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run data ingestion first: python -m services.ingestion.reliable_api_fetcher")
    # Load data from database (prioritize reliable sources)
    conn = sqlite3.connect('data/agrisage.db')
    
    # Load data from reliable tables only
    try:
        weather_df = pd.read_sql_query("SELECT * FROM reliable_weather", conn)
        print(f"Using reliable weather data: {len(weather_df)} records")
    except Exception as e:
        print(f"Error loading weather data: {e}")
        weather_df = pd.DataFrame()
    
    try:
        soil_df = pd.read_sql_query("SELECT * FROM reliable_soil", conn)
        print(f"Using reliable soil data: {len(soil_df)} records")
    except Exception as e:
        print(f"Error loading soil data: {e}")
        soil_df = pd.DataFrame()
    
    try:
        market_df = pd.read_sql_query("SELECT * FROM reliable_markets", conn)
        print(f"Using reliable market data: {len(market_df)} records")
    except Exception as e:
        print(f"Error loading market data: {e}")
        market_df = pd.DataFrame()
    
    documents = []
    metadatas = []
    ids = []
    
    # Weather forecast data (handle both reliable and fallback schemas)
    for index, row in weather_df.iterrows():
        doc_id = row.get('id', index)
        district = row['district']
        date = row.get('date', row.get('forecast_date', 'unknown'))
        precip = row.get('precip_prob', 0)
        max_temp = row.get('max_temp', 25)
        min_temp = row.get('min_temp', 15)
        source = row.get('source', 'weather_forecast')
        
        text = f"Weather forecast for {district} on {date}: {precip}% chance of precipitation, max temp {max_temp}°C, min temp {min_temp}°C"
        if 'description' in row and row['description']:
            text += f", conditions: {row['description']}"
        if 'rainfall' in row and row['rainfall']:
            text += f", rainfall: {row['rainfall']}mm"
            
        documents.append(text)
        metadatas.append({
            "source": source,
            "row_id": str(doc_id),
            "district": district,
            "date": date,
            "type": "weather"
        })
        ids.append(f"weather_{doc_id}")
    
    # Soil health data (handle both reliable and fallback schemas)
    for index, row in soil_df.iterrows():
        doc_id = row.get('id', index)
        district = row['district']
        
        if 'village' in row:
            # Fallback schema
            village = row['village']
            pH = row['pH']
            N = row.get('N', row.get('nitrogen', 0))
            P = row.get('P', 0)
            K = row.get('K', 0)
            organic_carbon = row.get('organic_carbon', 0)
            text = f"Soil analysis for {village}, {district}: pH {pH}, Nitrogen {N}, Phosphorus {P}, Potassium {K}, Organic Carbon {organic_carbon}%"
        else:
            # Reliable schema
            pH = row['pH']
            nitrogen = row.get('nitrogen', 0)
            organic_carbon = row.get('organic_carbon', 0)
            sand_percent = row.get('sand_percent', 0)
            clay_percent = row.get('clay_percent', 0)
            text = f"Soil analysis for {district}: pH {pH:.1f}, Nitrogen {nitrogen:.1f}%, Organic Carbon {organic_carbon:.1f}%, Sand {sand_percent}%, Clay {clay_percent}%"
        
        documents.append(text)
        metadatas.append({
            "source": row.get('source', 'soil_card'),
            "row_id": str(doc_id),
            "district": district,
            "type": "soil"
        })
        ids.append(f"soil_{doc_id}")
    
    # Market prices data (handle both reliable and fallback schemas)
    for index, row in market_df.iterrows():
        doc_id = row.get('id', index)
        date = row['date']
        commodity = row['commodity']
        mandi = row.get('mandi', 'Unknown Mandi')
        price = row['price']
        district = row.get('district', mandi.split()[0] if mandi else "unknown")
        
        text = f"Market price for {commodity} at {mandi} on {date}: ₹{price} per unit"
        documents.append(text)
        metadatas.append({
            "source": row.get('source', 'market_prices'),
            "row_id": str(doc_id),
            "district": district,
            "date": date,
            "type": "market"
        })
        ids.append(f"market_{doc_id}")
    
    # eNAM trade data (if available)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='enam_trades';
    """)
    if cursor.fetchone():
        cursor.execute("""
            SELECT rowid, date, commodity, mandi, trade_volume, price 
            FROM enam_trades
        """)
        for row in cursor.fetchall():
            doc_id, date, commodity, mandi, volume, price = row
            text = f"eNAM trade for {commodity} at {mandi} on {date}: {volume} units traded at ₹{price}"
            documents.append(text)
            metadatas.append({
                "source": "enam_trades",
                "row_id": str(doc_id),
                "commodity": commodity,
                "mandi": mandi,
                "date": date,
                "type": "trade"
            })
            ids.append(f"enam_{doc_id}")
    
    conn.close()
    return documents, metadatas, ids

def build_chroma_index():
    """Build Chroma vector database index"""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading data from database...")
    documents, metadatas, ids = load_data_from_db()
    
    if not documents:
        raise ValueError("No documents found in database. Run ETL first.")
    
    print(f"Found {len(documents)} documents to index")
    
    # Initialize Chroma client
    chroma_path = Path("services/rag/chroma_db")
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("agri")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="agri",
        metadata={"description": "AgriSage agricultural knowledge base"}
    )
    
    print("Generating embeddings and building index...")
    
    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Generate embeddings
        embeddings = model.encode(batch_docs).tolist()
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
        print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    print(f"Index built successfully with {len(documents)} documents")
    return collection

def test_index():
    """Test the built index with sample queries"""
    print("\nTesting index with sample queries...")
    
    chroma_path = Path("services/rag/chroma_db")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("agri")
    
    test_queries = [
        "When should I irrigate wheat in Roorkee?",
        "What is the soil pH in my area?",
        "Current market price for rice",
        "Weather forecast for tomorrow"
    ]
    
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\nQuery: {query}")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"  {i+1}. [{meta['source']}] {doc[:100]}... (distance: {dist:.3f})")

def main():
    """Main function to build and test index"""
    try:
        collection = build_chroma_index()
        test_index()
        print("\nVector index built successfully!")
        print("Next step: Start the API server with 'uvicorn services.api.app:app --reload'")
        
    except Exception as e:
        print(f"Error building index: {e}")
        print("Make sure to run data ingestion first: python -m services.ingestion.reliable_api_fetcher")

if __name__ == "__main__":
    main()
