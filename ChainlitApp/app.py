import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("en.env")
BASE_URL = os.environ.get('AZURE_OPENAI_API_BASE')
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
os.environ['OPENAI_API_BASE']=BASE_URL
os.environ['OPENAI_API_KEY']=API_KEY
CHAT_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')

chat_llm = AzureChatOpenAI(
            temperature=0,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=CHAT_DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            streaming=True
        )
from langchain.memory import ConversationBufferMemory

#os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"
'''
template = """Question: {question}

Answer: Let's think step by step."""
'''
template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {question}
Chatbot:"""

@cl.langchain_factory(use_async=True)
def factory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(template=template, input_variables=["question","chat_history"])
    
    llm_chain = LLMChain(prompt=prompt, llm=chat_llm, verbose=True,memory=memory)

    return llm_chain
