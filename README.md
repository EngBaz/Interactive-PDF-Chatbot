# Hybrid RAG with LangChain, FAISS and OpenAI GPT-4

This project presents an example of a Q&A assistant using LangChain, OpenAI GPT-4, FAISS vectorstore, and Streamlit.
The assistant can handle conversational context and assist in answering questions related to an uploaded document.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Implementation](#Implementation)

## Introduction
The project showcases the implementation of a custom question-answering RAG system that leverages LangChain, an open-source framework, to interact with users in a conversational manner. The assistant answers questions related to a specific uploaded document and uses GPT-4 for natural language understanding and generation.

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/Hybrid-RAG-System</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>

4. Obtain an API key from OpenAI and Cohere AI. Store the Cohere API key in a <code>.env</code> file with the corresponsding name <code>COHERE_API_KEY</code>.
    
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

A <code>hybrid search</code> system was developed that uses <code>FAISS (Facebook AI Similarity Search)</code> as a vector database. This approach combines traditional <code>keyword-based</code> search with <code>vector-based</code> search that captures contextual meaning. By merging these methods, the system can deliver more accurate and relevant results. LangChain's <code>EnsembleRetriever</code> tool is used to effectively integrate these two search techniques. A <code>Cohere Rerank</code> model is used after the hybrid search to further improve the ranking of search results.

1. <code>FAISS (Facebook AI Similarity Search):</code> FAISS is an efficient library designed to search for similar elements in a large collection of vectors. It is characterized by optimized indexing and search algorithms when processing large data sets. In this context, FAISS stores vector embeddings, i.e. mathematical representations of data that capture the semantic meaning and relationships between them.
   
2. <code>Keyword search:</code> Conventional search methods focus on finding exact keywords or phrases to retrieve results. This method is very effective for structured or specific queries, but can miss the broader context and meaning behind the words, especially if the phrases or synonyms are different.

3. <code>Vector search:</code> Unlike keyword search, vector search is based on embeddings that represent words, phrases or entire documents as vectors in a multidimensional space. By capturing the semantic relationships between the data, vector search finds results based on contextual similarity, even if the exact keywords are not used.

4. <code>Hybrid search:</code> This system combines both approaches — keyword search provides precision when exact terms match, while vector search adds context by retrieving results based on meaning and semantics. The hybrid approach aims to improve relevance and handle different types of search queries more robustly.

5. <code>LangChain’s EnsembleRetriever:</code> The EnsembleRetriever is a special tool within LangChain that enables the combination of several search mechanisms (e.g. keyword and vector search). It intelligently combines the strengths of both methods and ensures that the search results are more comprehensive and contextualized.

6. <code>Cohere Rerank Model:</code> The Cohere Rerank Model is a machine learning-based tool that further improves the ranking of search results. After the hybrid search system has retrieved the initial results (from both keyword and vector searches), the Cohere model is applied to re-rank these results based on relevance. By analyzing the semantic relationship between the search query and the search results, it assigns a higher score to those that best match the intent of the search query. This final reordering ensures that the most contextually appropriate and useful information is prioritized, improving the overall quality of results.

## References

[1] https://arxiv.org/pdf/2408.05141

[2] https://arxiv.org/abs/2409.07691

[3] https://python.langchain.com/v0.2/docs/introduction/

[4] https://python.langchain.com/docs/tutorials/rag/

[5] https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps


