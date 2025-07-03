from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("QnA Bot with LLAMA3 (via Langchain + Ollama)")
input_text = st.text_input("Ask me anything:")

# Initialize Ollama LLM
llm = Ollama(model="llama3.2:latest")  # or "llama3:latest"
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Process input
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": input_text})
        st.write("### Answer:")
        st.write(response)
