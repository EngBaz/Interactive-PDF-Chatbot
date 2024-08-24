# Conversational Agent With Langchain and OpenAI GPT-4

This end-to-end project presents an example of a Q&A conversational agent using Langchain, OpenAI GPT-4 and Streamlit.
The agent can handle conversational context and assist in answering questions related to an uploaded document.

## Introduction
The project showcases the implementation of a custom chat agent that leverages Langchain, an open-source framework, to interact with users in a conversational manner. The agent answers questios related to a specific uploaded document. This agent is powered by GPT-4 for natural language understanding and generation.

## Setup
1. Clone this repository to your local machine by running <code>git clone github.com/EngBaz/conversational-retrieval-agent.git</code>
    
2. Install the required dependencies by running <code>pip install -r requirements.txt</code>

3. Obtain an API key from OpenAI
    
4. Note that the project is built using OpenAI GPT4. Thus, it is necessary to have an OpenAI API. Otherwise, for the use of open-source LLMs on huggingface, import your model using the steps below.
    ```console
    
    $ pip install langchain huggingface_hub
    $ os.environ['HUGGINGFACE_API_TOKEN'] = 'your_hugging_face_api_token'
    $ llm = HuggingFaceHub(repo_id="model_name", model_kwargs={'temperature': 0.7, 'max_length': 64})
    ```

## Usage

To use the conversational agent:
1. In the terminal, rub the streamlit app: <code> streamlit run app.py </code>
2. Then, select the appropriate file format to upload
3. Upload your file
4. Write specific questions about the uploaded file
5. The agent will process the input and respond with relevant information 
   
