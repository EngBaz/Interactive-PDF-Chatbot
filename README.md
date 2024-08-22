# Q&A Conversational Agent

This end-to-end project presents an example of a Q&A chatbot for extracting information from an PDF document. To run this project, follow the below steps.

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
4. Please note that the project is built using OpenAI GPT-4. Thus, it is necessary to have an OpenAI API. Otherwise, if you want to use an open-source LLM, import your model using the steps below.
    ```console
    
    $ pip install langchain huggingface_hub
    $ os.environ['HUGGINGFACE_API_TOKEN'] = 'your_hugging_face_api_token'
    $ llm = HuggingFaceHub(repo_id="model_name", model_kwargs={'temperature': 0.7, 'max_lenght': 64})
    ```

