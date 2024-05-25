from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

load_dotenv(find_dotenv())
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
memory=ConversationBufferWindowMemory(k=5)
conversation = ConversationChain(llm=chat, memory=memory)
print(conversation.invoke("tell me a joke")["response"])