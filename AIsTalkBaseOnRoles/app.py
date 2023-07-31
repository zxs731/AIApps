
import json, ast
import os
import chainlit as cl
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import AIAgents as aiAgents

assistant_role_name = "Python Programmer"
user_role_name = "Stock Trader"



@cl.on_chat_start
async def main():
    global assistant_role_name
    global user_role_name
    res = await cl.AskUserMessage(content=f"What is Chatbot role? (After 10s default Chatbot role: [{assistant_role_name}] will be set.)", timeout=10).send()
    if res:
    	assistant_role_name=res['content']
    res2 = await cl.AskUserMessage(content=f"What is user role?(After 10s default user role: [{user_role_name}] will be set.)", timeout=10).send()
    if res2:
    	user_role_name=res2['content']

    await cl.Message(
            content=f"Please input a task! The chatbot:[{assistant_role_name}] will assist user [{user_role_name}] on this task implementation",
        ).send()

@cl.on_message
async def run_conversation(user_message: str):
    chat_turn_limit, n = 10, 0
    user_task = user_message #"Develop a trading bot for the stock market"
    assistant_agent,user_agent,assistant_msg,user_msg,specified_task=aiAgents.InitialAIAgents(a_role_name =assistant_role_name,u_role_name = user_role_name,u_task = user_task)
    print(f"Original task prompt:\n{user_task}\n")
    print(f"Specified task prompt:\n{specified_task}\n")
    await cl.Message(author="Chatbot", content=specified_task).send()
    
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)
        #print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
        await cl.Message(author="User", content=user_msg.content).send()
        
        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        #print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        await cl.Message(author="Chatbot", content=assistant_msg.content).send()
        if "<CAMEL_TASK_DONE>" in user_msg.content:
            break 


