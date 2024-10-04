# Advanced RAG with LangGraph, FAISS and Groq Llama 3

This project presents an example of a Q&A assistant using LangChain, Groq Llama 3, FAISS vectorstore, and Streamlit.
The assistant can handle conversational context and assist in answering questions related to an uploaded document.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Implementation](#Implementation)

## Introduction
The project showcases the implementation of a custom question-answering RAG system that leverages LangGraph, an open-source framework, to interact with users in a conversational manner. The assistant answers questions related to a specific uploaded document and uses advanced RAG techniques such as corrective-RAG.

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/Hybrid-RAG-System</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>

4. Obtain an API key from OpenAI, Cohere AI and Groq. Store the APIs in a <code>.env</code> file as follows:
    ```console
    
    $ OPENAI_API_KEY="your api key"
    $ GROQ_API_KEY="your api key"
    $ COHERE_API_KEY="your api key"
    ```

## Usage

To use the conversational agent:
1. In the terminal, run the streamlit app: <code> streamlit run main.py </code>
2. Select the appropriate format of your file 
3. Upload your file
4. Write a specific question about the uploaded file
5. The agent will process the input and respond with relevant information
6. 

## References

[1] Hybrid RAG: https://arxiv.org/pdf/2408.05141

[2] https://arxiv.org/abs/2409.07691

[3] https://python.langchain.com/v0.2/docs/introduction/

[4] https://python.langchain.com/docs/tutorials/rag/

[5] https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps


