#!/usr/bin/env python3
"""
AgriSage Streamlit Web Interface
Modern farmer-friendly chatbot UI with real-time agricultural advice
"""
import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "services"))

from rag.pipeline import ask, get_health, initialize

st.set_page_config(
    page_title="AgriSage - AI Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initialize RAG pipeline once per Streamlit session."""
    initialize()
    return True


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def run_agrisage_query(query: str, location: str = "Roorkee") -> dict:
    """Run a query through the in-process RAG pipeline."""
    load_pipeline()
    full_query = (
        f"{query} in {location}"
        if location and location.lower() not in query.lower()
        else query
    )

    try:
        result = ask(
            question=full_query,
            location=location,
            user_id="streamlit_user",
        )
        return {
            "response": result.get("answer", "No response"),
            "sources": [p.get("source", "Unknown") for p in result.get("provenance", [])],
            "confidence": result.get("confidence", 0.0),
            "escalate": result.get("escalate", False),
            "fallback_used": result.get("fallback_used", False),
            "provenance": result.get("provenance", []),
        }
    except Exception as exc:
        return {
            "response": f"❌ Error: {exc}",
            "sources": [],
            "confidence": 0.0,
            "escalate": False,
            "fallback_used": False,
            "provenance": [],
        }


def append_system_message(result: dict):
    st.session_state.messages.append({
        "content": result["response"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "escalate": result.get("escalate", False),
        "fallback_used": result.get("fallback_used", False),
        "provenance": result.get("provenance", []),
        "timestamp": datetime.now().strftime("%H:%M"),
    })


def display_message(message: dict, is_user: bool = False):
    css_class = "farmer-query" if is_user else "system-response"
    icon = "👨‍🌾" if is_user else "🤖"

    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {'You' if is_user else 'AgriSage'}:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

        if not is_user and (message.get("sources") or message.get("escalate")):
            with st.expander("📚 Response Details"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if message.get("sources"):
                        st.write("**Sources:**")
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.write(f"{i}. {source}")
                    if message.get("escalate"):
                        st.warning("Escalated to human expert")
                    if message.get("fallback_used"):
                        st.info("Fallback rules engine used")

                with col2:
                    st.metric("Confidence", f"{message.get('confidence', 0.0):.1%}")


def main():
    init_session_state()
    load_pipeline()

    st.markdown("""
    <div class="main-header">
        <h1>🌾 AgriSage - AI Agricultural Assistant</h1>
        <p>Get real-time weather forecasts, soil data, and farming advice</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")

        location = st.selectbox(
            "📍 Your Location",
            ["Roorkee", "Haridwar", "Dehradun", "Rishikesh", "Pauri Garhwal", "Other"],
            index=0,
        )

        if location == "Other":
            location = st.text_input("Enter your location:", placeholder="e.g., Haldwani")

        st.header("🚀 Quick Queries")
        quick_queries = [
            "Weather forecast for next 3 days",
            "Will it rain tomorrow in Roorkee?",
            "What is the soil pH in my area?",
            "Soil preparation for maize",
            "Best time to plant mustard",
        ]

        for query in quick_queries:
            if st.button(query, key=f"quick_{query}"):
                st.session_state.messages.append({
                    "content": query,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })
                with st.spinner("🤔 AgriSage is thinking..."):
                    result = run_agrisage_query(query, location)
                append_system_message(result)
                st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat with AgriSage")

        with st.container():
            for i, message in enumerate(st.session_state.messages):
                display_message(message, is_user=i % 2 == 0)

        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])

            with col_input:
                user_input = st.text_input(
                    "Ask AgriSage anything about farming:",
                    placeholder="e.g., What's the weather like? What is the soil pH?",
                    label_visibility="collapsed",
                )

            with col_send:
                send_button = st.form_submit_button("Send 📤", use_container_width=True)

            if send_button and user_input:
                st.session_state.messages.append({
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })
                with st.spinner("🤔 AgriSage is analyzing your query..."):
                    result = run_agrisage_query(user_input, location)
                append_system_message(result)
                st.rerun()

    with col2:
        st.header("📊 System Status")

        try:
            health_data = get_health()
            st.success("✅ Pipeline Ready")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Database", f"{health_data.get('database_records', 'N/A')} records")
            st.metric("Vector Index", f"{health_data.get('vector_documents', 'N/A')} docs")
            st.metric("Gemini", "Configured" if health_data.get("gemini_configured") else "Missing")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            st.warning("⚠️ Pipeline status unknown")

        st.header("💡 Usage Tips")
        st.info("""
        **Try asking:**
        - "Weather in Roorkee tomorrow"
        - "What is the soil pH in my area?"
        - "What's the nitrogen content in Haridwar?"
        - "Best time to plant mustard"
        """)


if __name__ == "__main__":
    main()
