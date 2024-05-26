from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain.chains import RetrievalQA

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
memory=ConversationBufferWindowMemory(ai_prefix="AI Assistant")
template = """
 Your are an AI assistant for scalable capital. Your should assist customers with questions they have.

{context}

Question: {question}
Answer here:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
conversation = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff", chain_type_kwargs={"prompt": prompt})
query = "Are joint accounts offered?"
print(conversation.invoke(query))