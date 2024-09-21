import os
import streamlit as st

from langchain_openai import ChatOpenAI
from rag_utils import process_file_and_answer

from dotenv import load_dotenv
load_dotenv()

# Set API key for Cohere
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

    
st.set_page_config(
    page_title="Chatbot",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
    )

with st.sidebar:
        
    OPENAI_API_KEY = st.text_input(":blue[Enter Your OPENAI API Key:]",
                                   placeholder="Paste your OpenAI API key here (sk-...)",
                                   type="password",
                                   )
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
    max_new_tokens = st.number_input("Select a max token value:", min_value=1, max_value=8000, value=1000)
        
    temperature = st.number_input("Select a temperature value:", min_value=0.0, max_value=1.0, value=0.00)
    
    selected_format = st.selectbox(label="Select file format:", options=["...", "pdf", "txt"])

    uploaded_file = st.file_uploader("Upload a file:", type=[selected_format])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

if OPENAI_API_KEY:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, openai_api_key=OPENAI_API_KEY, max_tokens=max_new_tokens)
            
    st.title("RAG Chat Assistant🤖")
        
    if uploaded_file:
        st.session_state.messages = []
        process_file_and_answer(uploaded_file, selected_format, llm)
            
    else:
        st.warning("Please upload a file to continue.")
        
