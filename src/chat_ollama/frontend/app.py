import streamlit as st
import ollama
from chat_ollama.ollama_utils import chat_model
from chat_ollama.constants import *

if "messages" not in st.session_state:
    st.session_state["messages"] = START_MESSAGES
if "model" not in st.session_state:
    st.session_state["model"] = DEFAULT_MODEL

st.title("Chat Ollama")
st.write(f"Using model: {st.session_state['model']}")
if st.button("Clear Chat"):
    st.session_state["messages"] = START_MESSAGES

if prompt := st.chat_input("Talk to your favorite ollama model ..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Write messages to UI
    for message in st.session_state["messages"][1:]:
        st.chat_message(message["role"]).write(message["content"])

    with st.chat_message("ai"):
        model_response = st.write_stream(
            chat_model(st.session_state["model"], st.session_state["messages"])
        )

    st.session_state["messages"].append({"role": "assistant", "content": model_response})


with st.sidebar:
    st.write("**Select a model:**")
    st.session_state["model"] = st.selectbox(
        label="Available Models", 
        options=[model.model for model in ollama.list()["models"]]
    )
