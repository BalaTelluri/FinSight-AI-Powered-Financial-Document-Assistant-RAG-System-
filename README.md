# 🏦 FinSight — AI-Powered Financial Document Assistant

> A **Retrieval-Augmented Generation (RAG)** application that enables users to query complex insurance and banking documents in plain English — with precise, source-cited answers in under 2 seconds.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Metrics & Evaluation](#-metrics--evaluation)
- [Interview Highlights](#-interview-highlights)
- [Author](#-author)

---

## 🔍 Overview

**FinSight** is a GenAI-powered financial document assistant built using the RAG (Retrieval-Augmented Generation) architecture. It allows users to ask natural language questions about insurance policies and banking products — and receive accurate, document-grounded answers with source citations.

The system separates documents into **4 categories** (Health Insurance, Car Insurance, Banking, Home Insurance), each with its own FAISS vector index, ensuring answers never mix content across domains.

---

## 🎯 Problem Statement

Financial documents like insurance policies and loan agreements are:

- ❌ 50–100 pages long
- ❌ Written in complex legal language
- ❌ Difficult for customers to navigate
- ❌ Leading to millions of unnecessary customer service calls

**FinSight solves this** by letting users upload their actual policy documents and ask questions in plain English — getting instant, accurate answers grounded in the real document content.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Spell Checker  │  ← Auto-corrects typos using Groq LLM
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Query Expansion │  ← Rewrites question for better retrieval
└─────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│           Category Router                   │
│  Health │  Car  │  Banking  │  Home         │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────┐
│  FAISS Search   │  ← Semantic similarity search (k=5 chunks)
│  (per category) │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Groq LLM       │  ← Llama 3.3 70B — generates precise answer
│  (Llama 3.3 70B)│
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  RAGAS Eval     │  ← Faithfulness, Relevancy, Precision scores
└─────────────────┘
      │
      ▼
Answer + Source Document + Metrics
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 Conversational Chat | Natural chat interface with category-based navigation and quick-reply buttons |
| 🔍 Category-Separated Search | 4 independent FAISS indexes — no cross-domain answer contamination |
| 🧠 Query Expansion | LLM rewrites user question before retrieval for higher accuracy |
| 🔤 Spell Correction | Auto-detects and corrects typos in financial terminology |
| 📄 Multi-Format Support | PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), CSV, TXT |
| 📊 Live Metrics Dashboard | Real-time response time, retrieval confidence, answer found rate, RAGAS scores |
| 🎯 RAGAS Evaluation | Faithfulness, Answer Relevancy, Context Precision per response |
| ⚡ Sub-2s Response | Groq-hosted Llama 3.3 70B — 30x faster than local CPU inference |
| 🔒 Strict Answer Control | Refuses to hallucinate — only answers from document content |
| 📎 Source Citations | Every answer shows which document and section it came from |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Llama 3.3 70B via Groq API | Answer generation |
| **Embeddings** | `all-mpnet-base-v2` (HuggingFace) | Text-to-vector conversion |
| **Vector Store** | FAISS | Semantic similarity search |
| **RAG Framework** | LangChain | Pipeline orchestration |
| **Evaluation** | RAGAS | Faithfulness, relevancy, precision |
| **UI** | Streamlit | Conversational chat interface |
| **PDF Parsing** | pypdf | Extract text from PDFs |
| **Word Parsing** | python-docx | Extract text from .docx files |
| **Excel Parsing** | openpyxl | Extract text from .xlsx files |
| **PPT Parsing** | python-pptx | Extract text from .pptx files |
| **Spell Check** | Groq LLM + symspellpy | Query correction |
| **Environment** | python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
finsight/
├── app.py                    # Main Streamlit application
├── requirements.txt          # All dependencies
├── .env                      # API keys (not committed to Git)
├── .env.example              # Template for environment setup
├── .gitignore
├── README.md
│
├── utils/
│   ├── rag_pipeline.py       # FAISS indexing, retrieval, LLM chain
│   ├── document_loader.py    # Multi-format document parser & chunker
│   ├── spell_check.py        # Query correction module
│   └── ragas_eval.py         # RAGAS evaluation metrics
│
└── data/
    ├── health/               # Health insurance PDFs
    ├── car/                  # Car insurance PDFs
    ├── banking/              # Banking & loan PDFs
    └── home/                 # Home insurance PDFs
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/finsight.git
cd finsight

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and paste your Groq API key:
# GROQ_API_KEY=your_key_here

# 5. Add your documents
# Place PDFs in the correct subfolder:
# data/health/    ← health insurance documents
# data/car/       ← car insurance documents
# data/banking/   ← banking/loan documents
# data/home/      ← home insurance documents

# 6. Run the app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📊 Metrics & Evaluation

FinSight tracks both **operational** and **RAG quality** metrics in real time:

### Operational Metrics (Live Dashboard)

| Metric | Description |
|---|---|
| Response Time | Seconds from question to answer |
| Retrieval Confidence | FAISS similarity score (0–100%) |
| Answer Found Rate | % of questions answered from documents |
| Questions by Category | Usage distribution across 4 domains |

### RAGAS Quality Metrics (per response)

| Metric | What it measures | Target |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the document? (no hallucination) | > 0.80 |
| **Answer Relevancy** | Does the answer address the actual question? | > 0.75 |
| **Context Precision** | Are the retrieved chunks relevant to the question? | > 0.70 |

---

## 💼 Interview Highlights

**One-line description:**
> "Built FinSight — a RAG system using LangChain, FAISS and Llama 3.3 70B that enables natural language querying of financial documents, with category-separated vector indexes, RAGAS evaluation and sub-2 second response times via Groq API."

**Key technical decisions:**

- Used `all-mpnet-base-v2` over `all-MiniLM-L6-v2` for 15–20% better retrieval accuracy
- Separate FAISS index per category eliminates cross-domain answer contamination
- Query expansion via LLM rewrites the question before retrieval, improving context recall
- Temperature=0 on the LLM ensures deterministic, precise answers with no randomness
- Real regulatory documents (IPID/SECCI) used — same format as production enterprise systems

---

## 👤 Author

**Bala Sai Kiran Reddy**  
ML Engineer | Data Scientist | Stuttgart, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/bala-sai-kiran-reddy-telluri-055370250/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/BalaTelluri)

---

*Built as part of a GenAI portfolio to demonstrate production-grade RAG architecture for financial document intelligence.*
