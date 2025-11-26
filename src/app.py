import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# --- Configuration --- 
VECTOR_DB_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

# --- RAG Pipeline Setup (No changes made here) ---
@st.cache_resource
def setup_rag_pipeline():
    """Initializes and returns the RAG chain using modern LangChain patterns."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"FATAL: Vector Store setup failed. Did you run 'python ingest.py'? Error: {e}")
        return None, None 

    llm = OllamaLLM(model=LLM_MODEL)
    template = """
You are a helpful assistant that answers questions strictly based on the given publication data.
If the answer is not in the data, you MUST reply with this exact phrase:
"I'm sorry, this information is not present in the publication."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- HANDLER FUNCTION TO SET STATE AND TRIGGER RERUN ---
def set_question(question):
    """Sets the question in session state and triggers a safe rerun."""
    st.session_state.query_input = question
    st.rerun()

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="🧠 NLP RAG Memory Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Content ---
st.title("🧠 NLP RAG Memory Assistant")

# --- Project Description ---
st.markdown("""
This Retrieval-Augmented Generation (RAG) system demonstrates advanced natural language processing capabilities, combining semantic search with large language model inference to provide **accurate, context-aware responses** grounded strictly in the provided publication data.
""")
st.markdown("---")


# --- Sidebar (Structured and Content Updated) ---
with st.sidebar:
    st.header("ℹ️ Project Details") 
    
    # 1. Models Used Section
    st.subheader("🤖 Models Used")
    st.markdown(f"""
    - **LLM:** `{LLM_MODEL}` (via Ollama)
    - **Embeddings:** `{EMBEDDING_MODEL.split('/')[-1]}`
    - **Vector Store:** FAISS
    """)
    st.divider()
    
    # 2. Technical Details Section
    st.subheader("⚙️ Technical Details")
    st.markdown("""
    - **Framework:** LangChain + Streamlit
    - **Retrieval Method:** Semantic Search
    - **Top-K Results:** 3 documents
    - **Chain Type:** LCEL Pipeline
    """)
    st.divider()
    
    # 3. Suggested Questions Section (FIXED: Using standard buttons for stability)
    st.subheader("💡 Suggested Questions")
    
    suggested_questions = [
        "What is the necessity of memory in RAG systems?",
        "What components are critical for efficient similarity search?",
        "How does the RecursiveCharacterTextSplitter maintain semantic coherence?",
        "What library provides efficient similarity search?"
    ]
    
    # Use standard Streamlit buttons (or st.markdown with a button inside a container)
    for i, question in enumerate(suggested_questions, 1):
        st.button(
            question, 
            on_click=set_question, 
            args=[question], 
            key=f"btn_{i}",
            use_container_width=True
        )


# --- Main Query Interface ---
st.subheader("🔍 Ask Your Question")

# Initialize RAG Pipeline
result = setup_rag_pipeline()

# --- Handle Query Input (Simplified) ---
# The button click will set st.session_state.query_input
query = st.text_input(
    "Enter your question based on the publication:",
    placeholder="Type your question here...",
    key="query_input" # This key is where the button will store the value
)


if result:
    rag_chain, retriever = result
    
    # Execute query if the input box is not empty
    if query:
        with st.spinner("🔎 Searching knowledge base and generating answer..."):
            try:
                # Invoke the RAG Chain
                answer = rag_chain.invoke(query)
                
                # Display answer in a nice box
                st.markdown("### 💬 Answer")
                st.success(answer)
                
                # Optionally show source documents
                with st.expander("📄 View Source Documents"):
                    # NOTE: Using retriever.invoke() as this was the last successful method used
                    docs = retriever.invoke(query) 
                    if docs:
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**📌 Source {i}:**")
                            st.code(doc.page_content, language="text")
                            if i < len(docs):
                                st.markdown("---")
                    else:
                        st.info("No source documents found.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred during query processing. Please ensure Ollama is running.\n\n**Error:** {e}")
    else:
        st.info("👆 Enter a question above or select a suggested question from the sidebar to get started.")
else:
    st.error("⚠️ Could not initialize RAG pipeline. Please check the error messages above and ensure you have run `python ingest.py` to create the vector store.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>Built with LangChain, Streamlit, and Ollama | Powered by FAISS Vector Search</p>
</div>
""", unsafe_allow_html=True)