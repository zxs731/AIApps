import os

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter


import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
load_dotenv("en.env")
BASE_URL = os.environ.get('AZURE_OPENAI_API_BASE')
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
os.environ['OPENAI_API_BASE']=BASE_URL
os.environ['OPENAI_API_KEY']=API_KEY
CHAT_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


embeddings = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT_NAME,
        chunk_size=1
    )

#docsearch = Chroma(persist_directory='mybooks', embedding_function=embeddings)




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        '''
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        '''    
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    docsearch=Chroma.from_documents(docs, embeddings, persist_directory="mybooks")
    docsearch.persist()
    return docsearch


@cl.langchain_factory(use_async=True)
async def langchain_factory():
    
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    
    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)
    
    chat_llm = AzureChatOpenAI(
            temperature=0,
            model_name="gpt-35-turbo-16k",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name='gpt35turbo16k',
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            
        )
    
    docsearch = Chroma(persist_directory='mybooks', embedding_function=embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history",     return_messages=True,output_key='answer')
    chain = ConversationalRetrievalChain.from_llm(chat_llm, docsearch.as_retriever(), memory=memory, return_source_documents=True)

    '''
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    '''
    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain

'''
@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []
    
    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"
    
    await cl.Message(content=answer, elements=source_elements).send()
'''

@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    source_elements_dict = {}
    source_elements = []
    found_sources=[]

    for idx, source in enumerate(res["source_documents"]):
        title = source.metadata["source"]

        if title not in source_elements_dict:
            source_elements_dict[title] = {
                "page_number": [source.metadata["page"]],
                "url": source.metadata["source"],
                "content":source.page_content,
            }

        else:
            source_elements_dict[title]["page_number"].append(source.metadata["page"])
        source_elements_dict[title]["content_"+str(source.metadata["page"])]=source.page_content
        # sort the page numbers
        #source_elements_dict[title]["page_number"].sort()

    for title, source in source_elements_dict.items():
        # create a string for the page numbers
        page_numbers = ", ".join([str(x) for x in source["page_number"]])
        text_for_source = f"Page Number(s): {page_numbers}\nURL: {source['url']}"
        source_elements.append(
            cl.Pdf(name="File", path=title)
        )
        found_sources.append("File")
        for pn in source["page_number"]:
            source_elements.append(
                cl.Text(name=str(pn), content=source["content_"+str(pn)])
            )
            found_sources.append(str(pn))
        
    if found_sources:
        answer+=f"\nSource:{', '.join(found_sources)}"
    else:
        answer+=f"\nNo source found."

    await cl.Message(content=answer, elements=source_elements).send()