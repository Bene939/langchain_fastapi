from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts.prompt import PromptTemplate

load_dotenv(find_dotenv())
groq = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
memory=ConversationBufferWindowMemory(k=5, ai_prefix="AI Assistant")
template = """
 The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:
"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(prompt=prompt, llm=groq, memory=memory)
print(conversation.invoke("tell me a joke")["response"])