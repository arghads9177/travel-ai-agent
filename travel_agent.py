import os
from dotenv import load_dotenv
import streamlit as st
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.agent import Agent, AgentKnowledge
from textwrap import dedent
from agno.utils.pprint import pprint_run_response

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define Knowledgebase for Travel Data
travel_knowledge_base = AgentKnowledge(
    vector_db=ChromaDb(
        collection="travel_data",
        path="./data/chroma_db",
        embedder=OpenAIEmbedder(),
        persistent_client=True
    )
)

# Define Travel Guide Agent
travel_agent = Agent(
    model=None,  # No LLM model is used here; we rely on the knowledge base
    knowledge=travel_knowledge_base,
    search_knowledge=True,
    description=dedent("""\
        You are a helpful Travel Guide AI Agent specialized in providing information about popular tourist destinations.
        Your task is to search the knowledge base and retrieve the most relevant travel information for the user’s query.

        Your answer style is:
        - Clear and informative
        - Engaging but professional
        - Answer must be specific to the question.\
    """),
    instructions=dedent("""\
        Follow these rules strictly:

        Search the knowledge base for travel information and provide the best match.
        If the information is not found in the knowledge base, respond with:
        "I don’t have travel information on that location."

        If the query is not related to travel (e.g., general questions or off-topic queries), politely respond with:
        "I'm here to assist with travel-related queries. Feel free to ask about destinations, attractions, or travel tips!"
        Stay polite, concise, and on-topic while assisting users.\
    """),
    markdown=True,
    debug_mode=True
)

# Streamlit UI
def main():
    st.set_page_config(page_title="🌍 Travel Guide AI Agent", page_icon="✈️")
    st.title("🌍 Travel Guide AI Agent")
    st.write("Welcome to the Travel Guide AI Agent! Ask me about popular destinations, attractions, or travel tips.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Ask me anything about travel...")

    if user_input:
        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate agent response
        response = travel_agent.run(user_input)

        # Add agent response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.write(response.content)

# Run the Streamlit app
if __name__ == "__main__":
    main()