import streamlit as st

# Load vectorstore and conversation chain from language_model module
from language_model import load_conversation_chain

# Setup chat
st.title('Scalable Capital FAQ Bot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load conversation chain
st.session_state["conversation"] = load_conversation_chain()

# React to user input
if user_prompt := st.chat_input("Do you offer joint accounts?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    response = st.session_state["conversation"].invoke(user_prompt)["answer"]
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
