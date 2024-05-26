from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

#setup env
load_dotenv(find_dotenv())

#setup vector database
loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
with open("faq_vectorstore.pkl", "wb")as f:
    pickle.dump(vectorstore, f)

#load db
with open("faq_vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

#setup model
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
template = """
 Your are an AI assistant for scalable capital. Your should assist customers with questions they have. Do not warn that you can't offer financial advice.

{context}

Question: {question}
Answer here:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
conversation = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vectorstore.as_retriever(), combine_docs_chain_kwargs={"prompt": prompt})

#setup chat
st.title('Scalable Capital FAQ Bot')
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.session_state["conversation"] = conversation

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