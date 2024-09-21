import time
import streamlit as st

from PyPDF2 import PdfReader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker 
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank


# Function to configure the retrieval with chat history
def configure_hybrid_search(data):
    """
    Configures a hybrid search mechanism that combines semantic and keyword-based retrieval,
    followed by reranking for optimized search results.
    
    Args:
        data: The input data that will be chunked and used for retrieval.

    Returns:
        compression_retriever: A retrieval system that combines hybrid search and reranking for improved search performance.
    """
    
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
    
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever,
    )
    
    return compression_retriever



def configure_rag_chain(retriever, llm):
    """
    Configures a Retrieval-Augmented Generation (RAG) chain that uses a retriever and a large language model
    to handle conversational question answering with session-specific history.

    Args:
        retriever: The document retriever responsible for fetching relevant context from a knowledge base.
        llm: The large language model used to generate answers to the user's questions.

    Returns:
        conversational_rag_chain: A RAG chain that handles input messages, retrieves relevant context,
        manages chat history, and generates responses based on both the retrieved context and prior conversations.
    """
    
    contextualize_system_prompt ="""
    
    Given a chat history and a user's last question that could refer to previous context or information 
    from the chat history, your task is to formulate a clear and stand-alone question or statement that 
    can be understood without reference to the previous conversation.

    If the user's question refers to something previously mentioned in the chat (e.g. follow-up questions, 
    vague allusions or clarifications), you need to refer back to the chat history, understand the specific 
    context the user is alluding to, and formulate an accurate, self-contained question or statement that 
    conveys the full meaning. This conversion is essential because the rephrased version will be used to query 
    a vector database for relevant information.

    Your goal is to ensure that the rephrased question or statement is as precise and specific as possible, 
    especially when it comes to follow-up questions or vague allusions, so that it retrieves the most relevant 
    information from the database.
    
    """
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
        
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_prompt,
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise and clear.\
    {context}
    """
        
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



def stream_data(response):
    
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)    



def process_file_and_answer(uploaded_file, file_format, llm):
    """
    Processes an uploaded file based on its format and sets up a Retrieval-Augmented Generation (RAG) chain 
    to answer questions related to the file.
    
    Args:
        uploaded_file: The file uploaded by the user.
        file_format: The format of the uploaded file (e.g., "pdf", "csv", "txt", or "py").
        llm: The large language model used for question-answering.
    """

    # Clear chat history when a new file is uploaded
    if uploaded_file:
        st.session_state.messages = []
        
    # Extract data depending on the format of the file
    if file_format == "pdf":
        
        pdf_reader = PdfReader(uploaded_file)
        data = "".join(page.extract_text() for page in pdf_reader.pages)
   
    elif file_format == "txt":
        
        data = uploaded_file.read().decode("utf-8")
            
    else:
        
        pass

    retriever = configure_hybrid_search(data)
    
    conversational_rag_chain = configure_rag_chain(retriever, llm)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Welcome message from assistant
    if not st.session_state.messages:
        welcome_message = "Hello there, how can I help you today with your document?👋"
        st.chat_message("assistant").write_stream(stream_data(welcome_message))
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    # Accept user input    
    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write_stream(stream_data(prompt))    
        
        with st.chat_message("assistant"):
            
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "session1"}},
                )["answer"]
            
            st.write_stream(stream_data(response))
        st.session_state.messages.append({"role": "assistant", "content": response})    


