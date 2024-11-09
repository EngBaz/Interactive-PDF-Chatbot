import os

from rag_utils import *

from langchain_groq import ChatGroq
from pydantic import (
    BaseModel,
    Field,
)
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from typing import (
    List,
)
from langgraph.graph import END, StateGraph
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

# Set up streamlit UI
st.set_page_config(
    page_title="Conversational Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initiate the sidebar of the streamlit application
with st.sidebar:
    st.header("Setup")

    file_format = st.selectbox(label="Select file format:", options=["...", "pdf", "txt", "csv"])

    uploaded_file = st.file_uploader("Upload a file:", type=[file_format])

    st.header("Model Settings")

    max_new_tokens = st.number_input("Select a max token value:", min_value=1, max_value=8000, value=1000)

    temperature = st.number_input("Select a temperature value:", min_value=0.0, max_value=2.0, value=0.00)

# Load the LLM from GROQ
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=temperature, max_tokens=max_new_tokens)

# Initiate Tavily web search tool
web_search_tool = TavilySearchResults(k=5)

# Load the retriever and the RAG chain
retriever = retrieve_documents(uploaded_file, file_format)
rag_chain = create_rag_chain(retriever, llm)


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

grader_system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grader_human = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system),
        ("human", grader_human)
    ]
)

retrieval_grader = grade_prompt | structured_llm_doc_grader


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")

    web_search_flag = state["web_search"]

    if web_search_flag == "Yes":

        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:

        print("---DECISION: GENERATE---")
        return "generate"


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke(
        {"input": question},
    )["answer"]

    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search_flag = "No"

    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)

        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search_flag = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search_flag}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents

    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


# Build the graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("websearch", web_search)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write_stream(stream_data(prompt))

    with st.chat_message("assistant"):
        inputs = {"question": prompt}
        output = app.invoke(inputs)

        st.write(stream_data(output["generation"]))
        st.session_state.messages.append({"role": "assistant", "content": output["generation"]})













