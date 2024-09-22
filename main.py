import os
import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI
from rag_utils import *

from dotenv import load_dotenv
load_dotenv()

# Set API key for Cohere
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

    
st.set_page_config(
    page_title="Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    )

def main():
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    with st.sidebar:
            
        st.header("SETUP")

        OPENAI_API_KEY = st.text_input(":blue[Enter Your OPENAI API Key:]",
                                        placeholder="Paste your OpenAI API key here (sk-...)",
                                        type="password",
                                        )
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
        file_format = st.selectbox(label="Select file format:", options=["...", "pdf", "txt", "csv"])

        uploaded_file = st.file_uploader("Upload a file:", type=[file_format])
            
        st.header("PARAMETERS")
                
        max_new_tokens = st.number_input("Select a max token value:", min_value=1, max_value=8000, value=1000)
                
        temperature = st.number_input("Select a temperature value:", min_value=0.0, max_value=1.0, value=0.00)
    
    if OPENAI_API_KEY:
            
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, openai_api_key=OPENAI_API_KEY, max_tokens=max_new_tokens)
        
        try:
            if not uploaded_file:
                
                st.warning("No file uploaded. Please upload a file to continue.")
                st.stop()

            elif uploaded_file.size == 0:
                
                st.info("The uploaded file is empty. Please upload a valid document.")
                st.stop()
        
            elif uploaded_file is not None:
                
                if file_format == "pdf":
                    
                    pdf_reader = PdfReader(uploaded_file)
                    data = "".join(page.extract_text() for page in pdf_reader.pages)
                    
                elif file_format == "txt":
            
                    data = uploaded_file.read().decode("utf-8")
                
                elif file_format == "csv":
                    
                    data = pd.read_csv(uploaded_file)
                    data = data.to_string(index=False)
                    
                else:
                    
                    st.warning("Unsupported file format. Please upload a valid document (PDF, TXT, or CSV).")
            
            else:
                pass
            
        except Exception as e:
            
            st.info(f"An error occurred the upload: {str(e)}")

        if 'data' in locals():
            
            retriever = configure_hybrid_search(data)

            conversational_rag_chain = configure_rag_chain(retriever, llm)
     
        if prompt := st.chat_input("Ask a question"):
        
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"): 
                
                st.write_stream(stream_data(prompt))    
            
            with st.chat_message("assistant"):
                
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "session1"}},
                    )["answer"]
                
                st.write_stream(stream_data(response))
                
            st.session_state.messages.append({"role": "assistant", "content": response})
        

if __name__ == "__main__":
    
    main()
    welcome_message()    
    
    
        
