import json, ast
import os
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv  
  
# 加载.env文件  
load_dotenv("en.env")  

os.environ["OPENAI_API_TYPE"] = os.environ["Azure_OPENAI_API_TYPE1"]
os.environ["OPENAI_API_BASE"] = os.environ["Azure_OPENAI_API_BASE1"]
os.environ["OPENAI_API_KEY"] = os.environ["Azure_OPENAI_API_KEY1"]
os.environ["OPENAI_API_VERSION"] = os.environ["Azure_OPENAI_API_VERSION1"]
BASE_URL=os.environ["OPENAI_API_BASE"]
API_KEY=os.environ["OPENAI_API_KEY"]


os.environ["BING_SUBSCRIPTION_KEY"] = os.environ["Azure_BING_SUBSCRIPTION_KEY1"]
os.environ["BING_SEARCH_URL"] = os.environ["BING_SEARCH_URL1"]
from langchain.schema.output_parser import StrOutputParser


CHAT_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.retrievers.web_research import WebResearchRetriever
import web_research as wr

def settings():

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore  
    embeddings_model = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT_NAME,
        chunk_size=16
    )
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

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

    # Search
    from langchain.utilities import BingSearchAPIWrapper
    search = BingSearchAPIWrapper()

    # Initialize 
    web_retriever = wr.BingWebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    return web_retriever, llm

@cl.on_chat_start
async def main():
    web_retriever, llm=settings()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(
            content=f"You can ask me everything!",
        ).send()
    
@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    qa_chain = cl.user_session.get("qa_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await qa_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print('`Answer:`\n\n' + res['answer'])
    print('`Sources:`\n\n' + res['sources'])
    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    if(len(res['sources'])>0):
        elements = [
            cl.Text(name="source", content=res['sources'], display="inline")
        ]
        await cl.Message(content=res["answer"], elements=elements).send()   
    else:
        await cl.Message(content=res["answer"]).send()   

