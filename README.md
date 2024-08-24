# Conversational Agent With Langchain and GPT4

This end-to-end project presents an example of a Q&A conversational agent using Langchain, OpenAI GPT4 and Streamlit.
The agent can handle conversational context and assist in answering questions related to an uploaded document.

## Introduction
The project showcases the implementation of a custom chat agent that leverages Langchain, an open-source framework, to interact with users in a conversational manner. The agent answers questios related to a specific uploaded document. This agent is powered by GPT-4 for natural language understanding and generation.
1. Create a virtual enviromnent 
    ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
    
2. Install dependencies 
    ```console
    $ pip install -r requirements.txt
    ```
    
3. Run the application using Streamlit
    ```console
    $ streamlit run app.py
    ```
4. Note that the project is built using OpenAI GPT4. Thus, it is necessary to have an OpenAI API. Otherwise, for the use of open-source LLMs on huggingface, import yourr model using the steps below.
    ```console
    
    $ pip install langchain huggingface_hub
    $ os.environ['HUGGINGFACE_API_TOKEN'] = 'your_hugging_face_api_token'
    $ llm = HuggingFaceHub(repo_id="model_name", model_kwargs={'temperature': 0.7, 'max_length': 64})
    ```

