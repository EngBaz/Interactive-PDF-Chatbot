import os
import streamlit as st
from langchain_openai import ChatOpenAI
from utilities import process_file_and_answer

from dotenv import load_dotenv
load_dotenv()

# Set API key for Cohere
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

def main():
    
    # Set up the page configuration of the Streamlit UI 
    st.set_page_config(
        page_title="Hybrid RAG",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded",
        )

    with st.sidebar:
        
        # Set a header for the sidebar
        st.header("Configuration!")
        
        # Set API key for OpenAI 
        OPENAI_API_KEY = st.text_input(":blue[Enter Your OPENAI API Key:]",
                                    placeholder="Paste your OpenAI API key here (sk-...)",
                                    type="password",
                                    )
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Select file format
        selected_format = st.selectbox(label="Select file format", options=["...", "pdf", "csv", "txt", "py"])
        
        # Upload a file
        uploaded_file = st.file_uploader("Upload a file", type=[selected_format])
    
    if OPENAI_API_KEY:
        
        # Set OpenAI LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
            
        # Set the title of the pages
        st.title("HybridGPT!🤖")
            
        if uploaded_file is not None:
            process_file_and_answer(uploaded_file, selected_format, llm)
        else:
            st.warning("Please upload a file to continue.")
        
        
if __name__ == "__main__":
    main()
