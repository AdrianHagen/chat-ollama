import streamlit as st
import ollama
from chat_ollama.ollama_utils import chat_model
from chat_ollama.constants import *
from chat_ollama.db_manager import *
from chat_ollama.frontend.frontend_utils import generate_chat_title

# Initialize database
init_database()

if "messages" not in st.session_state:
    st.session_state["messages"] = START_MESSAGES
if "model" not in st.session_state:
    st.session_state["model"] = DEFAULT_MODEL
if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = None

st.title("Chat Ollama")
st.write(f"Using model: {st.session_state['model']}")

# Display existing messages (from loaded chat or current conversation)
for message in st.session_state["messages"][1:]:  # Skip system message
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input("Talk to your favorite ollama model ..."):
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Create a new chat if this is the first user message
    if st.session_state["current_chat_id"] is None:
        # Generate chat title from first message
        chat_title = generate_chat_title(prompt, st.session_state["model"])
        # Create chat in database
        st.session_state["current_chat_id"] = create_chat(
            chat_title, 
            st.session_state["model"]
        )
    
    # Save user message to database
    add_message(st.session_state["current_chat_id"], "user", prompt)

    # Display the new user message
    st.chat_message("user").write(prompt)

    with st.chat_message("ai"):
        model_response = st.write_stream(
            chat_model(st.session_state["model"], st.session_state["messages"])
        )

    # Add assistant response to session state and database
    st.session_state["messages"].append({"role": "assistant", "content": model_response})
    add_message(st.session_state["current_chat_id"], "assistant", model_response)


with st.sidebar:
    st.write("**Select a model:**")
    st.session_state["model"] = st.selectbox(
        label="Available Models", 
        options=[model.model for model in ollama.list()["models"]]
    )
    
    st.divider()
    
    st.write("**Chat History:**")
    
    # Add "New Chat" button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state["messages"] = START_MESSAGES
        st.session_state["current_chat_id"] = None
        st.rerun()
    
    # Get all chats from database
    all_chats = get_all_chats()
    
    if all_chats:
        st.write(f"*{len(all_chats)} saved chat(s)*")
        
        # Display each chat as a button
        for chat in all_chats:
            # Highlight current chat
            button_label = chat["title"]
            if st.session_state["current_chat_id"] == chat["id"]:
                button_label = f"🔵 {button_label}"
            
            if st.button(button_label, key=f"chat_{chat['id']}", use_container_width=True):
                # Load the selected chat
                st.session_state["current_chat_id"] = chat["id"]
                st.session_state["model"] = chat["model"]
                
                # Load messages from database
                messages_from_db = get_messages(chat["id"])
                
                # Reconstruct messages list with system message
                st.session_state["messages"] = START_MESSAGES.copy()
                for msg in messages_from_db:
                    st.session_state["messages"].append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                st.rerun()
    else:
        st.write("*No saved chats yet*")
