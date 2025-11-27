# 🧠 NLP RAG Memory Assistant — Zero-Hallucination Knowledge Engine 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LCEL%20(LangChain%20Expression%20Language)-green)
![RAG](https://img.shields.io/badge/Task-RAG--Based%20Q&A-yellow)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(LLaMA--3)-red)
![Status](https://img.shields.io/badge/Project-Module%201%20Completed-brightgreen)

The **NLP RAG Memory Assistant** is a reliable, grounded Q&A system built as the Module 1 deliverable for the **Agentic AI Developer Certification**. It ensures all answers are fully evidence-based by enforcing strict RAG grounding rules using a custom knowledge base—making hallucinations nearly impossible.

This system leverages the performance and modularity of the **LangChain Expression Language (LCEL)**, running fully on-device using **Ollama (LLaMA-3)** with local semantic retrieval via FAISS.

---

## 🚀 Key Capabilities

This assistant is engineered to meet all Module 1 requirements:

- 🔒 **Zero Hallucinations:** The LLM is forced to answer **only** from retrieved context.
- ⚡ **Fast Retrieval:** FAISS-based vector search ensures efficient high-accuracy matching.
- 🧠 **Local LLM Execution:** Runs offline using Ollama and LLaMA-3 — no external API dependency.
- 🧩 **Modular LCEL Pipeline:** Clean, maintainable, and easily extendable codebase.

---

## 🛠 Architecture Overview

The system operates through a structured **3-stage RAG pipeline** orchestrated using LCEL.

### 1️⃣ Document Ingestion (`src/ingest.py`)

- Loads dataset from `data/memory_publication.json`
- Splits into semantic chunks (size: 800, overlap: 100)
- Builds embeddings using `all-MiniLM-L6-v2`
- Creates a persistent FAISS vector store

### 2️⃣ Retrieval + Prompt Building

- User query → FAISS retriever (`k=3`)
- Relevant context is formatted into a custom prompt template
- Non-matching queries trigger controlled fallback behavior

### 3️⃣ LLM Execution (`src/app.py`)

- The final structured prompt is sent to **Ollama (LLaMA-3)**
- The response is streamed to the **Streamlit UI**
- No answer outside retrieved content is allowed

---

## 💻 Installation & Setup

### 🧩 Prerequisites

- Python 3.10+
- Ollama installed and running
- LLaMA-3 model pulled locally

```bash
ollama pull llama3
```

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/maddiravi/nlp-rag-memory-assistant.git
cd nlp-rag-memory-assistant
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

### 3️⃣ Build Vector Store

```bash
python src/ingest.py
```

---

## 🏃 Running the Application

```bash
streamlit run src/app.py
```

### 🧭 How to Use

1. Open the Streamlit URL  
2. Ask any question related to the stored publication  
3. Receive:
   - A grounded answer OR  
   - A strict fallback response  

---

## 🧪 Validation & Behavior Rules

The system includes required behavior checks:

| Test Type | Example Input | Expected Output |
|----------|---------------|----------------|
| Retrieval Valid | _"What technique improves similarity search?"_ | FAISS-based answer grounded in text |
| Retrieval Fail | _"What is the capital of Australia?"_ | `"I'm sorry, this information is not present in the publication."` |

---

## 📂 Project Structure

```
nlp-rag-memory-assistant/
├── src/
│   ├── app.py               # Streamlit UI + LCEL RAG chain
│   └── ingest.py            # Data processing + FAISS builder
├── data/
│   └── memory_publication.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛡 License

This project is open-source and available under the **MIT License**.

---

## 🔖 Recommended Repo Tags

```
#RAG #LangChain #Python #Ollama #LLM #LLaMA3 #MemoryAssistant #FAISS #Retrieval #Streamlit #LocalInference #AgenticAI
```

