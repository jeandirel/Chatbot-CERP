import streamlit as st
from llm_chroma_main import query_llm  # Import function to query LLM
from constants import LLM_MODEL_PATH  # Import model path
import os
import uuid

import six
try:
    from six.moves import xrange 
except ImportError:
    zip = zip  # Python 3 : zip existe déjà
    xrange = range  # Python 3 : xrange est devenu range

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", layout="wide")

# Ensure user_id is stored in session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())  # Generate unique user ID

# st.title("💬 M3 AI Chatbot with")
st.title("🤖 Chatbot M3")
st.write("""Bonjour! Je suis Chatbot M3, un assistant intelligent capable de vous aider 
            avec des informations générales ainsi que des requêtes spécifiques sur notre base de connaissances M3.
            Posez-moi vos questions et je ferai de mon mieux pour y répondre! 😊""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""  # Store conversation history as a string

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("💬 Posez votre question:")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history += f"Utilisateur: {user_input}\n"  # Append to chat history

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get chatbot response
    with st.spinner("🔄 Génération de la réponse..."):
        bot_response = query_llm(user_input)  # Pass user_id automatically, chat history, Call LLM function

    # Add bot response to session history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.session_state.chat_history += f"Assistant: {bot_response}\n"  # Append bot response

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# Sidebar instructions
st.sidebar.title("ℹ️ Instructions")
st.sidebar.write("Ce chatbot utilise le modèle Mistral 7B avec récupération d’informations à partir des données de M3.")
