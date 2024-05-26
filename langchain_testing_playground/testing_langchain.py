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
import sys

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
print("AI Assistant: Hello! How can I assist you today? Type 'exit' to end the chat.")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("AI Assistant: Goodbye!")
        break
    try:
        response = conversation.invoke(query)
        print("AI Assistant:", response["answer"])
    except Exception as e:
        print("An error occurred:", e)