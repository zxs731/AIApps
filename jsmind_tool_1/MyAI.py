import chromadb
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain_decorators import GlobalSettings
import logging
from langchain_decorators import llm_prompt
from langchain.schema import HumanMessage

load_dotenv("en.env")
BASE_URL = os.environ.get('AZURE_OPENAI_API_BASE_001')
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY_001')
Version = os.environ.get('OPENAI_API_VERSION_001')
os.environ['OPENAI_API_BASE']=BASE_URL
os.environ['OPENAI_API_KEY']=API_KEY
CHAT_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')

print(BASE_URL)
chat_llm = AzureChatOpenAI(
            temperature=0.3,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=CHAT_DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            
        )
chat_llm_stream = AzureChatOpenAI(
            temperature=0.3,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=CHAT_DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            streaming=True,
        )
GlobalSettings.define_settings(
    default_llm=chat_llm, #this is default... can change it here globally
    default_streaming_llm=chat_llm_stream, #this is default... can change it here for all ... will be used for streaming
    logging_level=logging.CRITICAL, 
    print_prompt=False, 
    print_prompt_name=False
)
embeddings = OpenAIEmbeddings(
            deployment=EMBEDDING_DEPLOYMENT_NAME
        )

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#chat = ChatOpenAI(temperature=0)
@llm_prompt
def RelatedKeywords(text:str)->list:
    '''
    你是一个多学科专家，善于找到与给出的主题有关系的多个关键词列表。]
    主题:{text}
    '''
    pass

@llm_prompt
def GenerateQuestions(topic:str)->list:
    """你是一个多学科专家，善于对给定主题提问，善于提出他人不容易想到的问题。提出的问题也很全面。主题：{topic}
    """
    pass

@llm_prompt
def GetAnswer(topic:str)->list:
    '''问题:{topic}
    答案列表:'''
    pass
    

def GenerateMarkdown(topic):
    template = "你是一个信息处理助理，请从给定的上下文中找到热搜榜的内容，并将它们总结成思维导图markdown的格式输出， Note: only markdown format is needed no common"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "上下文Context：\n '{input_topic}'。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat_llm, prompt=chat_prompt)
    answer = chain.run(input_topic=topic)
    print(answer)


def GenerateHot():
    loader = WebBaseLoader("https://top.baidu.com/board?tab=realtime")
	#loader.requests_kwargs = {'verify':False}
    data = loader.load()
    template = "你是一个信息处理助理，请从给定的上下文中找到热搜榜的内容，并将它以如下格式输出，格式:['条目1','条目2',...]"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "上下文Context：\n '{input_topic}'。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat_llm, prompt=chat_prompt)
    answer = chain.run(input_topic=data[0].page_content[:2000])
    print(answer)
    return answer


def ReadBaidu(topic):
    loader = WebBaseLoader("https://www.baidu.com/s?wd="+topic)
	#loader.requests_kwargs = {'verify':False}
    data = loader.load()
    return data

def GenerateHotItems(topic):    
    t=ReadBaidu(topic)
    template = "你是一个信息处理助理，请从给定的上下文中总结出进一步思考的内容，并将它以如下格式输出，格式:['Question1','Question2',...]"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "上下文Context：\n '{input_topic}'。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat_llm, prompt=chat_prompt)
    answer = chain.run(input_topic=t[0].page_content[:2000])
    print(answer)
    return answer

@llm_prompt
def GenerateArticle(topic:str)->str:    
    '''你是一个写作助理，请从给定的上下文写一篇文章。上下文Context：\n '{topic}'。'''
    pass
