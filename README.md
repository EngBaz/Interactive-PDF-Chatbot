# Q&A Conversational Agent

This end-to-end project presents an example of a Q&A for extracting information from an PDF document. To run this project, follow the below steps.

1. Create a virtual enviromnent 
    ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
    
2. Install dependencies 
    ```console
    $ pip install -r requirements.txt
    ```
    
3. Run the application (app.py) with Streamlit
    ```console
    $ streamlit run app.py
    ```
Please note that the project is built using OpenAI GPT-4. Thus, it is necessary to have an OpenAI API. Otherwise, if you want to use an open-source LLM, import your model using the steps below.

