import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie

# --- UI CONFIG ---
st.set_page_config(page_title="AI PDF Assistant", page_icon="🧠", layout="centered")

# Custom CSS for smooth fades and better chat bubbles
st.markdown("""
    <style>
    .stChatMessage {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITIES ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a cool AI animation (Robot waving or brain pulsing)
lottie_ai = load_lottieurl("https://lottie.host/78229c29-399d-4340-bb53-3932731804c8/G2Q9fTzFzI.json")

# --- SIDEBAR ---
with st.sidebar:
    st_lottie(lottie_ai, height=150, key="sidebar_robot")
    st.title("Settings")
    st.caption("Powered by Groq & Pinecone")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT ---
st.title("🧠 PDF Knowledge Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Animation while thinking
        placeholder = st.empty()
        with placeholder.container():
            st_lottie(lottie_ai, height=100, key="loading_ai")
            st.caption("Searching through PDF chunks...")

        try:
            # 2. Call your FastAPI backend
            response = requests.post("http://chatbot-api:8000/chat", json={"question": prompt})
            full_response = response.json().get("answer", "Error: No response")
            
            # 3. Typewriter Effect (The "Pro" Look)
            placeholder.empty()
            message_placeholder = st.empty()
            typed_text = ""
            for chunk in full_response.split():
                typed_text += chunk + " "
                message_placeholder.markdown(typed_text + "▌")
                time.sleep(0.05) # Speed of typing
            message_placeholder.markdown(typed_text)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            placeholder.empty()
            st.error(f"Backend offline: {e}")
