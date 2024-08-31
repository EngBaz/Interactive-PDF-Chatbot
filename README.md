# RAG System with Langchain, FAISS and OpenAI GPT-4

This end-to-end project presents an example of a Q&A conversational agent using Langchain ecosystem, OpenAI GPT-4, FAISS vectorstore for RAG, and Streamlit.
The agent can handle conversational context and assist in answering questions related to an uploaded document.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Implementation](#Implementation)

## Introduction
The project showcases the implementation of a custom chat agent that leverages Langchain, an open-source framework, to interact with users in a conversational manner. The agent answers questions related to a specific uploaded document. This agent is powered by GPT-4 for natural language understanding and generation.

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/conversational-retrieval-agent.git</code>
    
2. Install the required dependencies by running <code>pip install -r requirements.txt</code>

3. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```

4. Obtain an API key from OpenAI and Cohere AI. Create a <code>.env</code> file and insert the API keys with the name <code>COHERE_API_KEY</code>
    
5. Note that the project is built using OpenAI GPT-4. Thus, it is necessary to have an OpenAI API key. Otherwise, for the use of open-source LLMs on huggingface, import your model using the steps below.
    ```console
    
    $ pip install langchain huggingface_hub
    $ os.environ['HUGGINGFACE_API_TOKEN'] = 'your_hugging_face_api_token'
    $ llm = HuggingFaceHub(repo_id="model_name", model_kwargs={'temperature': 0.7, 'max_length': 64})
    ```

## Usage

To use the conversational agent:
1. In the terminal, run the streamlit app: <code> streamlit run app.py </code>
2. Select the appropriate format of your file 
3. Upload your file
4. Write a specific question about the uploaded file
5. The agent will process the input and respond with relevant information

## Implementation

This section provides a brief summary of the techniques used to develop this project.

#### 1. Hybrid Search with Rerank Model

A hybrid search system was built using <code>FAISS</code> as a vector database. This system combines the precision of <code>keyword search</code>with the contextual understanding of <code>semantic search</code>, aiming for more accurate and comprehensive results. The Langchain <code>EnsembleRetriever</code> tool integrates these two search methods. After the initial search, a Cohere AI model is used to rerank the most relevant documents, further enhancing retrieval quality.

   
