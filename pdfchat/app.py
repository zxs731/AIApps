import gradio as gr
import os
import time

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter


from langchain.llms import OpenAI

from langchain.embeddings import OpenAIEmbeddings


from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("en.env")
BASE_URL = os.environ.get('AZURE_OPENAI_API_BASE')
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
os.environ['OPENAI_API_BASE']=BASE_URL
os.environ['OPENAI_API_KEY']=API_KEY
CHAT_DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')

llm = AzureChatOpenAI(
            temperature=0,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=CHAT_DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            
        )
embeddings = OpenAIEmbeddings(
            deployment=EMBEDDING_DEPLOYMENT_NAME
        )
db = Chroma(persist_directory='pdfvector', embedding_function=embeddings)
retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            return_source_documents=False)

        
def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, open_ai_key):
    global qa 
    print(pdf_doc.name)
    if False:
        os.environ['OPENAI_API_KEY'] = open_ai_key
        loader = OnlinePDFLoader(pdf_doc.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.5), 
            retriever=retriever, 
            return_source_documents=False)
        return "Ready"
    else:
        llm = AzureChatOpenAI(
            temperature=0,
            model_name="gpt-35-turbo",
            openai_api_base=BASE_URL,
            openai_api_version="2023-03-15-preview",
            deployment_name=CHAT_DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            
        )
        
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_doc.name)
        #loader = OnlinePDFLoader(pdf_doc.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(
            deployment=EMBEDDING_DEPLOYMENT_NAME
        )
        
        db = Chroma(persist_directory='pdfvector', embedding_function=embeddings)
        for i in range(0,len(texts)):
            db.add_texts([texts[i].page_content])
        db.persist()
        
        retriever = db.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            return_source_documents=False)
        return "Ready"
        

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    history[-1][1] = ""
    
    for character in response:     
        history[-1][1] += character
        #time.sleep(0.05)
        yield history
    

def infer(question, history):
    
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai)
        res.append(pair)
    
    chat_history = res
    #print(chat_history)
    query = question
    result = qa({"question": query, "chat_history": chat_history})
    #print(result)
    return result["answer"]

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column():
            openai_key = gr.Textbox(label="You OpenAI API key", type="password")
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(pdf_changes, inputs=[pdf_doc, openai_key], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot)

demo.queue()
demo.launch()
