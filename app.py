import os
from api import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings ⚙️")
    
    # Output format selection
    output_format = st.selectbox(
        'Select output format:',
        ('Paragraph', 'Bullet Points', 'Step-by-Step Guide', 'Table')
    )

    # Task complexity selection
    task_complexity = st.select_slider(
        'Task complexity:',
        options=['Simple', 'Moderate', 'Complex'],
        value='Moderate'
    )

    st.markdown("---")
    st.markdown("### Example Tasks")
    st.markdown("📝 Write a summary of climate change")
    st.markdown("🔍 Explain quantum physics")
    st.markdown("📊 Create a table of top AI tools")

# Main Content
st.title('🤖 AI Assistant')
st.markdown("### Your personal assistant for any task. Let's get started!")

# Task Input
user_input = st.text_input('What would you like me to do?')

# Loading spinner for better UX
if user_input:
    with st.spinner('Processing your request...'):
        
        # Task template
        task_template = PromptTemplate(
            input_variables=['task'],
            template='You are an AI assistant. Your task is to: {task}'
        )

        # Response template
        response_template = PromptTemplate(
            input_variables=['task', 'context', 'output_format', 'task_complexity'],
            template='''As an AI assistant, complete the following task: {task}
            Use this additional context if relevant: {context}
            Format your response as: {output_format}
            Treat this task as {task_complexity} in complexity.
            Your response:'''
        )

        # Memory
        task_memory = ConversationBufferMemory(input_key='task', memory_key='chat_history')
        response_memory = ConversationBufferMemory(input_key='task', memory_key='chat_history')

        # LangChain LLM setup
        llm = OpenAI(temperature=0.7)
        task_chain = LLMChain(llm=llm, prompt=task_template, verbose=True, output_key='task_description', memory=task_memory)
        response_chain = LLMChain(llm=llm, prompt=response_template, verbose=True, output_key='response', memory=response_memory)

        # Wikipedia API
        wiki = WikipediaAPIWrapper()

        # Run the task chain to get task description
        task_description = task_chain.run(user_input)
        
        # Fetch additional context from Wikipedia
        context = wiki.run(user_input)

        # Generate response using response chain
        response = response_chain.run(task=task_description, context=context, output_format=output_format, task_complexity=task_complexity)

        # Display task and response
        st.subheader("Task Description:")
        st.write(task_description)

        st.subheader("AI Response:")
        st.write(response)

        # Expander sections
        with st.expander('Task History 📝'):
            st.info(task_memory.buffer)

        with st.expander('Response History 🧠'):
            st.info(response_memory.buffer)

        with st.expander('Additional Context 🌐'):
            st.info(context)

#
