import streamlit as st
import time
import pandas as pd

from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


def retrieve_documents(uploaded_file, file_format):
    """
       This function handles the processing of an uploaded file (PDF, TXT, or CSV)
       and creates a document retriever capable of retrieving context-relevant
       documents for downstream tasks.

       Args:
           uploaded_file: The file uploaded by the user, which can be in PDF, TXT, or CSV format.
           file_format: A string indicating the format of the uploaded file, allowing the function to handle
           the file appropriately.

       Returns:
           retriever: combines both similarity-based and keyword-based retrieval methods, enhanced by contextual
           compression using a reranking model.
       """

    try:
        if not uploaded_file:
            welcome_message()
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

    semantic_chunker = SemanticChunker(
        OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile",
    )

    docs = semantic_chunker.create_documents([data])
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

    similarity_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[similarity_retriever, keyword_retriever],
                                           weights=[0.5, 0.5],
                                           )

    compressor = CohereRerank(model="rerank-english-v3.0")

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return retriever


def create_rag_chain(retriever, llm):
    """
     Creates a Retrieval-Augmented Generation (RAG) chain for answering questions by retrieving relevant
     context from a retriever and generating concise responses with a language model.

     Args:
         retriever: The document retriever used to fetch relevant context.
         llm: The language model responsible for generating responses.

     Returns:
         rag_chain: A RAG chain that integrates document retrieval with question answering.
     """

    qa_system = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def welcome_message():
    """
    Displays a welcome message to the user if there are no previous messages in the session state.This function
    introduces the assistant and prompts the user to enter an API key and upload a file.The message is only shown
    once at the start of a new session.

    Args:
        None

    Returns:
        None
    """

    if not st.session_state.messages:
        welcome_input = """

        Hello there! First, please specify the format of the file you'd like to upload. 
        Once you've uploaded your file, feel free to ask any questions about its content, and I'll provide 
        you with the relevant information. 👋

        """
        st.chat_message("assistant").write_stream(stream_data(welcome_input))
        st.session_state.messages.append({"role": "assistant", "content": welcome_input})


def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.03)

