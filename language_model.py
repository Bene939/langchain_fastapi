import os
import pickle
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Setup environment
load_dotenv(find_dotenv())

VECTORSTORE_PATH = "faq_vectorstore.pkl"

# Setup vector database
def setup_vectorstore():
    loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

# Load vectorstore from file
def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        setup_vectorstore()
    with open(VECTORSTORE_PATH, "rb") as f:
        return pickle.load(f)

# Setup and return conversation chain
def load_conversation_chain():
    vectorstore = load_vectorstore()
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    template = """
    You are an AI assistant for Scalable Capital. You should assist customers with questions they have. Do not warn that you can't offer financial advice.
    If the user greets you introduce yourself in one sentence.

    {context}

    Question: {question}
    Answer here:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    conversation = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vectorstore.as_retriever(), combine_docs_chain_kwargs={"prompt": prompt})
    return conversation
