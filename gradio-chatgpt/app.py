import gradio as gr
import random
import time
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from dotenv import load_dotenv
load_dotenv("en.env")
from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent


BASE_URL = os.environ.get('OPENAI_API_BASE')
API_KEY = os.environ.get('OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
DEPLOYMENT_NAME = "chatgpt0301"


with open("custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS) as demo:
    chatbot = gr.Chatbot().style(height="100%")
    msg = gr.Textbox(label="You")
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        llm = AzureChatOpenAI(
            temperature=0.5,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name="chatgpt0301",
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            verbose=True
        )
        '''
        llm=AzureOpenAI(temperature=0,deployment_name="davinci003", model_name="text-davinci-003")
        '''
        #llm=OpenAI(temperature=0)
        #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        his_messages=[]
        for i in history[0:-1]:
            his_messages.append(HumanMessage(content=i[0]))
            his_messages.append(AIMessage(content=i[1]))
        
        his_messages.append(HumanMessage(content=history[-1][0]))
        bot_message = llm(his_messages)
        
        print(bot_message.content)
        history[-1][1] = ""
        
        for character in bot_message.content:
            history[-1][1] += character
            #time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()
