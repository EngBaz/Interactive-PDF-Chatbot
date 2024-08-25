import streamlit as st
import os
import pandas as pd
import time

from utilities import get_apikey
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader 


# Set up the page of the Streamlit UI 
st.set_page_config(
    page_title="Q&A Agent",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
    )
 
 
# Set API key for OpenAI 
OPENAI_API_KEY = get_apikey()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Function to configure the retrieval and the RAG chain with chat history
def configure_rag_chain(loader):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
      
    texts = text_splitter.split_text(loader)
    vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    contextualize_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )
        
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
        )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""
        
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}
        
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
            return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )

    return conversational_rag_chain


# Function to stream the output with Streamlit
def stream_data():
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)


if OPENAI_API_KEY:
    
    # Set OpenAI LLM and embeddings
    llm = ChatOpenAI(model="gpt-4",temperature=0, openai_api_key=OPENAI_API_KEY)
    
    # Set the configuration for streamlit UI
    st.title("Q&A Conversational Agent!🤖")
    selected_format = st.selectbox(label="Select file format", options=["...", ".pdf", ".csv"])

    # Upload a CSV or PDF file
    uploaded_file = st.file_uploader("Upload a file!")

    if uploaded_file is not None:
        
        if selected_format == ".pdf":
            
            pdf_reader = PdfReader(uploaded_file)
            data = ""
            for page in pdf_reader.pages:
                data += page.extract_text()        
            conversational_rag_chain = configure_rag_chain(data)
            question = st.text_input("Ask any question!")
            submit = st.button("Submit!")
        
            if submit:
                response = conversational_rag_chain.invoke(
                    {"input": question},
                    config={
                        "configurable": {"session_id": "session1"}
                        },
                    )["answer"]
                st.write_stream(stream_data)
            
            
        elif selected_format == ".csv":
            
            df = pd.read_csv(uploaded_file)
            df_string = df.to_string()
            conversational_rag_chain = configure_rag_chain(df_string)
            question = st.text_input("Ask any question!")
            submit = st.button("Submit!")
        
            if submit:
                response = conversational_rag_chain.invoke(
                    {"input": question},
                        config={
                            "configurable": {"session_id": "session1"}
                            },
                        )["answer"]
                st.write_stream(stream_data)
                
        else:
            st.error("Please select a correct file format!")
    












  
  
      