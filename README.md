# Chatbot using Langchain + FastAPI + Streamlit
This is a simple chatbot.
It uses Streamlit as a front end.
Langchain is used to connect an LLM (accessed via Groq API) to a knowledge base (stored as a vector DB).
FastAPI is used as a way for the front end to communicate with the model.

By doing so the API could easily be extended for further functionality e.g. calling computer vision model to interpret a picture the user uploaded.
Langchain could also be used to add further functionality e.g. by first using an llm to identify the user's intent after which either the chatbot keeps assisting the user or a human is called to resolve the situation

### Starting the App
1. Add "GROQ_API_KEY" to .env file
2. python fastapi_backend.py
3. streamlit run streamlit_frontend.py

![chatbot_example_conversation (1)](https://github.com/user-attachments/assets/b0ff9cd0-9cf9-4d1e-b3c3-cf0f2aec47da)
