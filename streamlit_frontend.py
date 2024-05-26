import streamlit as st
import requests

# Setup chat
st.title('Scalable Capital FAQ Bot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_prompt := st.chat_input("Do you offer joint accounts?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    url = f"http://127.0.0.1:8000/assistant_response?question={user_prompt}"

    response = requests.get(url).json()
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
