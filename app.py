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
st.title('AutoGPT - AI Research Assistant')
prompt = st.text_input('Enter your topic:') 

# Add output format selection
output_format = st.selectbox(
    'Select output format:',
    ('Essay', 'Presentation Slides', 'Q&A Format')
)

# Add research depth selection
research_depth = st.select_slider(
    'Select research depth:',
    options=['Brief Overview', 'Detailed Analysis', 'Expert Level']
)

title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a research title about {topic}'
)

content_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'output_format', 'research_depth'], 
    template='Create a {output_format} about {title} with a {research_depth}. Use this wikipedia research: {wikipedia_research}. Format the output appropriately for {output_format}.'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
content_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True, output_key='content', memory=content_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    content = content_chain.run(title=title, wikipedia_research=wiki_research, output_format=output_format, research_depth=research_depth)

    st.write(title) 
    st.write(content) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Content History'): 
        st.info(content_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
