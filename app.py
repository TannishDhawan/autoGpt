import os 
from api import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('AI Assistant')
user_input = st.text_input('What would you like me to do?') 

# Output format selection
output_format = st.selectbox(
    'Select output format:',
    ('Paragraph', 'Bullet Points', 'Step-by-Step Guide', 'Table')
)

# Task complexity selection
task_complexity = st.select_slider(
    'Select task complexity:',
    options=['Simple', 'Moderate', 'Complex']
)

task_template = PromptTemplate(
    input_variables = ['task'],
    template='You are an AI assistant. Your task is to: {task}'
)

response_template = PromptTemplate(
    input_variables = ['task', 'context', 'output_format', 'task_complexity'],
    template='''As an AI assistant, complete the following task: {task}
    
    Use this additional context if relevant: {context}
    
    Format your response as: {output_format}
    
    Treat this task as {task_complexity} in complexity.
    
    Your response:'''
)

# Memory 
task_memory = ConversationBufferMemory(input_key='task', memory_key='chat_history')
response_memory = ConversationBufferMemory(input_key='task', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.7) 
task_chain = LLMChain(llm=llm, prompt=task_template, verbose=True, output_key='task_description', memory=task_memory)
response_chain = LLMChain(llm=llm, prompt=response_template, verbose=True, output_key='response', memory=response_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a user input
if user_input: 
    task_description = task_chain.run(user_input)
    context = wiki.run(user_input)
    response = response_chain.run(task=task_description, context=context, output_format=output_format, task_complexity=task_complexity)

    st.write("Task:")
    st.write(task_description) 
    st.write("Response:")
    st.write(response) 

    with st.expander('Task History'): 
        st.info(task_memory.buffer)

    with st.expander('Response History'): 
        st.info(response_memory.buffer)

    with st.expander('Additional Context'): 
        st.info(context)
