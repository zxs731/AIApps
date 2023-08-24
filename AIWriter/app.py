import openai 
from dotenv import load_dotenv  
import os
# 加载.env文件  
load_dotenv("en.env")  

os.environ["OPENAI_API_TYPE"] = os.environ["Azure_OPENAI_API_TYPE1"]
os.environ["OPENAI_API_BASE"] = os.environ["Azure_OPENAI_API_BASE1"]
os.environ["OPENAI_API_KEY"] = os.environ["Azure_OPENAI_API_KEY1"]
os.environ["OPENAI_API_VERSION"] = os.environ["Azure_OPENAI_API_VERSION1"]
BASE_URL=os.environ["OPENAI_API_BASE"]
API_KEY=os.environ["OPENAI_API_KEY"]


# LLM
from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(
    temperature=0,
    model_name="gpt-35-turbo-16k",
    openai_api_base=BASE_URL,
    openai_api_version="2023-07-01-preview",
    deployment_name="gpt35turbo16k",
    openai_api_key=API_KEY,
    openai_api_type = "azure",
    streaming=True
)

from langchain.prompts import PromptTemplate
header1_template=PromptTemplate(
    input_variables=["topic"],
    template="写一个虚构小说的标题，内容是关于‘{topic}’"
)

header2_template=PromptTemplate(
    input_variables=["title","topic"],
    template="参考如下虚构小说的背景，为标题是'{title}'的文章写目录的章节标题。\n虚构小说的背景：\n{topic}\n\n目录:",
)
count_template=PromptTemplate(
    input_variables=["toc"],
    template="下面有几个章节标题：\n{toc}\n\n标题的数目为(只要数字):",
)
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

title_memory = ConversationBufferMemory(memory_key="history", input_key="topic")
section_memory = ConversationBufferMemory(memory_key="history", input_key="title")
count_memory = ConversationBufferMemory(memory_key="history", input_key="toc")
content_memory = ConversationBufferMemory(memory_key="history", input_key="no")
from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
content_template = """结合这个虚拟小说的背景和目录，根据选择的章节名称完成段落内容的编写。

背景:
{topic}

目录:
{toc}

选择的章节:第{no}章
"""

prompt = PromptTemplate(input_variables=["topic","toc", "no"], template=content_template)

llm_title_chain = LLMChain(llm=llm, prompt=header1_template, memory=title_memory,verbose=True,output_key="title")
llm_section_chain = LLMChain(llm=llm, prompt=header2_template, memory=section_memory,verbose=True,output_key="toc")
llm_count_chain = LLMChain(llm=llm, prompt=count_template, memory=count_memory,verbose=True,output_key="no")
from langchain.chains import SequentialChain

llm_content_chain = LLMChain(llm=llm, prompt=prompt, memory=content_memory,verbose=True,output_key="content")
            
import streamlit as st
from thinking import StreamlitCallbackHandler
st.chat_message("assistant").write("请提供简要故事背景，包括人、事、物。以及故事梗概。AI帮你实现各章节的撰写。")
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            
            seqChain = SequentialChain(chains=[llm_title_chain, llm_section_chain,llm_count_chain],
                input_variables=["topic"],
                # Here we return multiple variables
                output_variables=["title", "toc","no"],
                verbose=True,
                callbacks=[st_cb]
            )
            a=seqChain({"topic":prompt})
            print(a)
            print(a["toc"])
            
            msgList=[]
            msgList.append(a["title"])
            msgList.append(a["toc"])
            
            for i in range(eval(a["no"])):
                st_cb1 = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True,collapse_completed_thoughts=True)
                c = llm_content_chain.run({"no":i+1,"toc":a["toc"],"title":a["title"],"topic":prompt},callbacks=[st_cb1])
                msgList.append(c)
            
            for msg in msgList:
                st.markdown(msg)
                
            
            
    
