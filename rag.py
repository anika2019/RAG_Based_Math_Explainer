import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Configuration & Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2026 Model Constants
EMBEDDING_MODEL = "models/gemini-embedding-001"
# DeepSeek-R1 is the "Math King" for reasoning in 2026
LLM_MODEL = "llama-3.3-70b-versatile" 
VECTOR_STORE_PATH = Path(__file__).parent / "resources/math_vector_store"

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

def initialize_math_rag():
    """Initializes the Embedding and LLM components."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    
    # We use a lower temperature (0.1) for Math to ensure accuracy over creativity
    llm = ChatGroq(
        model=LLM_MODEL, 
        temperature=0.1, 
        api_key=GROQ_API_KEY
    )
    return embeddings, llm

def build_knowledge_base(urls, embeddings):
    """Scrapes UP Board/NCERT data and saves it locally."""
    print("📚 Loading Math resources...")
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()

    # Math-Specific Splitter: Higher overlap to keep formulas together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_STORE_PATH))
    print(f"✅ Knowledge base saved to {VECTOR_STORE_PATH}")
    return vector_store

def get_math_help(vector_store, llm, student_query):
    """Retrieves context and generates a step-by-step math explanation."""

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Custom System Prompt for U.P. Board Students
    system_prompt = (
        "You are a helpful U.P. Board Math Assistant. "
        "Use the provided context to explain mathematical concepts step-by-step. "
        "If the question is in Hindi, answer in Hindi. If English, answer in English. "
        "Always show the formula used before solving. "
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the RAG Chain (LCEL) without requiring `langchain.chains`
    retriever = vector_store.as_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n🤔 Thinking about: {student_query}")
    return rag_chain.invoke(student_query)

if __name__ == "__main__":
    # Suggested U.P. Board / NCERT Math Resources
    math_urls = [
        "https://ncert.nic.in/textbook.php?mhh1=1-15", # NCERT Class 10 Math (Hindi)
        "https://www.teachoo.com/subjects/cbse-maths/class-10th/chapter-4-quadratic-equations/solutions/",
    ]

    # Initialize
    emb, llm = initialize_math_rag()

    # Step 1: Ingest (Run this once, then you can comment it out)
    v_store = build_knowledge_base(math_urls, emb)

    # Step 2: Query
    query = "द्विघात समीकरण (Quadratic Equation) को हल करने का सूत्र क्या है?"
    answer = get_math_help(v_store, llm, query)
    
    print("\n--- 🎓 ASSISTANT RESPONSE ---")
    print(answer)