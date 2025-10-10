#!/usr/bin/env python3
"""
AgriSage Streamlit Web Interface
Modern farmer-friendly chatbot UI with real-time agricultural advice
"""
import streamlit as st
import requests
import json
from datetime import datetime
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure Streamlit page
st.set_page_config(
    page_title="AgriSage - AI Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for farmer-friendly design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f8f9fa;
    }
    .farmer-query {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .system-response {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_base_url' not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"

def call_agrisage_api(query: str, location: str = "Roorkee") -> dict:
    """Call AgriSage API with farmer query"""
    try:
        # Format query for API
        full_query = f"{query} in {location}" if location and location.lower() not in query.lower() else query
        
        response = requests.post(
            f"{st.session_state.api_base_url}/ask",
            json={
                "question": full_query,
                "location": location,
                "user_id": "streamlit_user"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            api_result = response.json()
            # Convert API format to expected format
            return {
                "response": api_result.get("answer", "No response"),
                "sources": [p.get("source", "Unknown") for p in api_result.get("provenance", [])],
                "confidence": api_result.get("confidence", 0.0)
            }
        else:
            return {
                "response": f"❌ API Error: {response.status_code}",
                "sources": [],
                "confidence": 0.0
            }
    
    except requests.exceptions.ConnectionError:
        return {
            "response": "🔌 Cannot connect to AgriSage API. Please ensure the backend is running on http://localhost:8000",
            "sources": [],
            "confidence": 0.0
        }
    except Exception as e:
        return {
            "response": f"❌ Error: {str(e)}",
            "sources": [],
            "confidence": 0.0
        }

def display_message(message: dict, is_user: bool = False):
    """Display chat message with styling"""
    css_class = "farmer-query" if is_user else "system-response"
    icon = "👨‍🌾" if is_user else "🤖"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {'You' if is_user else 'AgriSage'}:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources and confidence for system responses
        if not is_user and 'sources' in message and message['sources']:
            with st.expander("📚 Data Sources & Confidence"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Sources:**")
                    for i, source in enumerate(message['sources'][:3], 1):
                        st.write(f"{i}. {source}")
                
                with col2:
                    confidence = message.get('confidence', 0.0)
                    st.metric("Confidence", f"{confidence:.1%}")

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AgriSage - AI Agricultural Assistant</h1>
        <p>Get real-time weather forecasts, market prices, and farming advice</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Location selection
        location = st.selectbox(
            "📍 Your Location",
            ["Roorkee", "Dehradun", "Haridwar", "Rishikesh", "Pauri Garhwal", "Other"],
            index=0
        )
        
        if location == "Other":
            location = st.text_input("Enter your location:", placeholder="e.g., Haldwani")
        
        # API endpoint
        api_url = st.text_input(
            "🔗 API Endpoint", 
            value=st.session_state.api_base_url,
            help="AgriSage backend API URL"
        )
        st.session_state.api_base_url = api_url
        
        # Quick query buttons
        st.header("🚀 Quick Queries")
        
        quick_queries = [
            "Weather forecast for next 3 days",
            "Will it rain tomorrow in Roorkee?",
            "What is the soil pH in my area?",
            "Soil preparation for maize",
            "Best time to plant mustard"
        ]
        
        for query in quick_queries:
            if st.button(query, key=f"quick_{query}"):
                # Add to chat
                st.session_state.messages.append({
                    "content": query,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Get AI response
                with st.spinner("🤔 AgriSage is thinking..."):
                    result = call_agrisage_api(query, location)
                
                st.session_state.messages.append({
                    "content": result["response"],
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.0),
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat with AgriSage")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                is_user = i % 2 == 0  # Alternate between user and system
                display_message(message, is_user)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Ask AgriSage anything about farming:",
                    placeholder="e.g., What's the weather like? Rice prices? Best fertilizer for wheat?",
                    label_visibility="collapsed"
                )
            
            with col_send:
                send_button = st.form_submit_button("Send 📤", use_container_width=True)
            
            if send_button and user_input:
                # Add user message
                st.session_state.messages.append({
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Get AI response
                with st.spinner("🤔 AgriSage is analyzing your query..."):
                    result = call_agrisage_api(user_input, location)
                
                # Add system response
                st.session_state.messages.append({
                    "content": result["response"],
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.0),
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                st.rerun()
    
    with col2:
        st.header("📊 System Status")
        
        # API health check
        try:
            health_response = requests.get(f"{st.session_state.api_base_url}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API Online")
                health_data = health_response.json()
                
                # Display system metrics
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Database", f"{health_data.get('database_records', 'N/A')} records")
                st.metric("Vector Index", f"{health_data.get('vector_documents', 'N/A')} docs")
                st.metric("Uptime", health_data.get('uptime', 'N/A'))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ API Offline")
        except:
            st.warning("⚠️ API Status Unknown")
        
        # Usage tips
        st.header("💡 Usage Tips")
        st.info("""
        **Try asking:**
        - "Weather in Roorkee tomorrow"
        - "Is it good weather for harvesting?"
        - "What is the soil pH in my area?"
        - "Soil preparation for maize"
        - "Best time to plant mustard"
        """)
        
        # Language support
        st.header("🌐 Language")
        st.info("Currently supports English. Hindi support coming soon!")

if __name__ == "__main__":
    main()
