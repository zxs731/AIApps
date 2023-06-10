import gradio as gr
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv("en.env")
BASE_URL = os.environ.get('Azure_ChatGPT_URL')
API_KEY = os.environ.get('Azure_OPENAI_API_KEY')

with open("custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

    
def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    history[-1][1] = ""

    headers = {
      'Content-Type': 'application/json',
      'api-key': API_KEY
    }
    
    messages = []
    for i in history:
        messages.extend([{"role": "user", "content": i[0]}])
        if i[1]:
            messages.extend([{"role": "assistant", "content": i[1]}])

    payload = json.dumps({"messages": messages, "stream": True})
    response = requests.post(BASE_URL, headers=headers, data=payload)
    '''
    for chunk in response.iter_lines():
        if chunk:
            json_str = chunk.decode('utf-8').replace('data:', '')
            if json_str!=" [DONE]":
                json_data = json.loads(json_str)
                delta = json_data['choices'][0].get("delta")
                if delta:
                    value = delta.get("content")
                    if value:
                        history[-1][1] += value
                        yield history
    '''
    for chunk in response.iter_lines():
        if chunk:
            json_str = chunk.decode('utf-8').replace('data:', '')
            if json_str!=" [DONE]":
                json_data = json.loads(json_str)
                reply = json_data['choices'][0].get("delta",[]).get("content",None)
                print(reply)
                if reply:
                    history[-1][1] += reply
                    yield history

with gr.Blocks(css=customCSS) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="You")
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0")  
    # 注意：gradio启动项目后默认地址为127.0.0.1；使用docker部署需要将地址修改为0.0.0.0，否则会导致地址访问错误
    # 默认端口为7860，如需更改可在launch()中设置server_port=7000
                                      