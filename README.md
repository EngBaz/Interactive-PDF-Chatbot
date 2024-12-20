# Interactive PDF Chatbot with LangGraph, FAISS, Groq LLaMA 3, and Streamlit

![brAIcht](images/pdf_chatbot.png)

## Introduction
The project showcases the implementation of a custom question-answering RAG system that leverages LangGraph, an open-source framework, to interact with users in a conversational manner. The assistant answers questions related to a specific uploaded document and uses advanced RAG techniques such as corrective-RAG.

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/Interactive-PDF-Chatbot.git</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>

4. Obtain an API key from OpenAI, Cohere AI and Groq. Store the APIs in a <code>.env</code> file as follows:
    ```console
    $ GROQ_API_KEY="your api key"
    $ COHERE_API_KEY="your api key"
    $ GOOGLE_API_KEY="your api key"
    $ TAVILY_API_KEY="your api key"
    ```

## Usage

To use the conversational agent:
1. In the terminal, run the streamlit app: <code> streamlit run main.py </code>
2. Select the appropriate format of your file 
3. Upload your file
4. Write a specific question about the uploaded file
5. The agent will process the input and respond with relevant information

## References

[1] Hybrid RAG: https://arxiv.org/pdf/2408.05141

[2] Rerankers: https://arxiv.org/abs/2409.07691

[3] Corrective-RAG: https://arxiv.org/abs/2401.15884

[4] https://python.langchain.com/docs/tutorials/rag/

[5] https://github.com/mistralai/cookbook/tree/main/third_party/langchain


