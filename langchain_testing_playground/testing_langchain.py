from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts.prompt import PromptTemplate

load_dotenv(find_dotenv())
groq = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
memory=ConversationBufferWindowMemory(ai_prefix="AI Assistant")
template = """
 Your are an AI investment advisor. Answer any questions the user might have truthfully. Don't ask questions.

Current conversation:
{history}
Human: {input}
AI Assistant:
"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(prompt=prompt, llm=groq, memory=memory)
print(conversation.invoke("i want to invest. can you recommend an etf for me?")["response"])