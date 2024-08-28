# Conversational Agent with Langchain and OpenAI GPT-4

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

4. Obtain an API key from OpenAI
    
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

## Description

This section provides a brief summary of the techniques used to develop this project.

#### 1. Hybrid Search

Hybrid search is a search method that combines the strengths of traditional keyword-based search and semantic search. It utilises the precision of keyword search with the contextual understanding and relevance of semantic search. By incorporating both approaches, hybrid search aims to deliver more accurate and comprehensive search results.

- Keyword Search

BM25 is a ranking algorithm that calculates how relevant a document is to a search query. It's an extension of th the TF-IDF method and generates a sparse vector. BM25 is used to improve search results by ranking documents based on their relevance to the user's query.

- Semantic Search

Semantic search is a more advanced way to find information. Unlike keyword search, which only looks for matching words, semantic search understands the meaning behind the words. This leads to better results. To do this, it measures how similar the search query is to the documents using scores. In this project, <code>cosine similarity</code> is used for this measurement.

This project combines semantic and keyword search techniques using FAISS as a vectorstore. The Langchain <code>EnsembleRetriever</code> tool is used to integrate these two approaches, providing more accurate search results.

   
