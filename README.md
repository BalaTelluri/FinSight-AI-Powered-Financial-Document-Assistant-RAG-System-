# 🏦 FinSight — AI-Powered Financial Document Assistant

> A production-ready **Retrieval-Augmented Generation (RAG)** application that enables users to query complex insurance and banking documents in plain English — with precise, source-cited answers in under 2 seconds.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup Project Environment](#-setup-project-environment)
- [Running the App](#-running-the-app)
- [Environment Variables](#-environment-variables)
- [Metrics & Evaluation](#-metrics--evaluation)
- [Author](#-author)

---

## 🔍 Overview

**FinSight** is a GenAI-powered financial document assistant built using the RAG (Retrieval-Augmented Generation) architecture. It allows users to ask natural language questions about insurance policies and banking products — and receive accurate, document-grounded answers with source citations.

The system separates documents into **4 categories** (Health Insurance, Car Insurance, Banking, Home Insurance), each with its own FAISS vector index, ensuring answers never mix content across domains.

**Target:** Answer user questions grounded strictly in uploaded financial documents

**Input features passed to RAG pipeline:**
```
category selection, user question, spell-corrected query, expanded query
```

---

## 🎯 Problem Statement

Financial documents like insurance policies and loan agreements are:

- ❌ 50–100 pages long
- ❌ Written in complex legal language
- ❌ Difficult for customers to navigate
- ❌ Leading to millions of unnecessary customer service calls

**FinSight solves this** by letting users ask questions in plain English and getting instant, accurate answers grounded directly in the real document content.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│    Spell Checker    │  ← Auto-corrects typos using Groq LLM
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Query Expansion   │  ← Rewrites question for better retrieval
└─────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│            Category Router               │
│  Health  │  Car  │  Banking  │  Home     │
└──────────────────────────────────────────┘
      │
      ▼
┌─────────────────────┐
│  FAISS Vector Search│  ← Semantic similarity search (k=5 chunks)
│   (per category)    │     all-mpnet-base-v2 embeddings
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Groq LLM          │  ← Llama 3.3 70B — generates precise answer
│   (Llama 3.3 70B)   │     Temperature = 0
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   RAGAS Evaluation  │  ← Faithfulness · Relevancy · Precision
└─────────────────────┘
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
| 📊 Live Metrics Dashboard | Real-time response time, retrieval confidence, answer found rate |
| 🎯 RAGAS Evaluation | Faithfulness, Answer Relevancy, Context Precision per response |
| ⚡ Sub-2s Response | Groq-hosted Llama 3.3 70B — 30x faster than local CPU inference |
| 🔒 Strict Answer Control | Refuses to hallucinate — only answers from document content |
| 📎 Source Citations | Every answer shows which document it came from |

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


## ⚙️ Setup Project Environment

### One-time setup

**Install Python 3.10+**

Link: https://www.python.org/downloads/

To check Python version:
```bash
python -V
```

> **Note:** Make sure Python is added to your system PATH (environment variables)

**Install pip** (comes with Python — verify with):
```bash
pip --version
```

**Clone the repository**
```bash
git clone https://github.com/yourusername/finsight.git
```

**Change directory into the repo**
```bash
cd finsight
```

**Create virtual environment**
```bash
python3 -m venv venv
```

**Activate virtual environment**

On Mac/Linux:
```bash
source venv/bin/activate
```

On Windows (cmd):
```bash
venv\Scripts\activate.bat
```

On Windows (Git Bash):
```bash
source venv/Scripts/activate
```

> You should see `(venv)` at the start of your terminal line — this confirms the environment is active.

**Install project requirements**
```bash
pip install -r requirements.txt
```

> **Note:** First install downloads HuggingFace embedding models (~500MB). Requires good internet connection. This only happens once.

> [!CAUTION]
> Do NOT run `pip freeze > requirements.txt` — it will overwrite the existing requirements file.

**How to add a new package**
```bash
pip install <package-name>
```

Then add the package name manually to `requirements.txt`.

Python Package Index official link: https://pypi.org/

---

## 🚀 Running the App

**Check your current working directory**
```bash
pwd
# output: /Users/yourname/finsight
```

**Make sure virtual environment is active**
```bash
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate.bat   # Windows
```

**Run the Streamlit app**
```bash
streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

> **Note:** Port 8501 is Streamlit's default. If it is already in use, run:
> ```bash
> streamlit run app.py --server.port 8502
> ```

---

## 🔑 Environment Variables

Create a `.env` file in the root of the project:

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

**How to get a free Groq API key:**

1. Go to https://console.groq.com
2. Sign up with Google or email (free)
3. Click **API Keys** in the left menu
4. Click **Create API Key**
5. Copy the key (looks like: `gsk_xxxxxxxxxxxxxxxxxxxx`)
6. Paste it into your `.env` file

> [!CAUTION]
> Never paste your API key directly in the terminal or share it publicly. Always store it in the `.env` file only.

> [!IMPORTANT]
> The `.env` file is listed in `.gitignore` — it will NOT be pushed to GitHub. This keeps your key private.

---

## 📂 Adding Your Documents

Place your PDF documents into the correct subfolder inside `data/`:

```
data/
├── health/     ← health insurance PDFs go here
├── car/        ← car insurance PDFs go here
├── banking/    ← banking / loan PDFs go here
└── home/       ← home insurance PDFs go here
```

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.csv`, `.txt`

> **Note:** Documents must be text-based PDFs (not scanned images). To verify — open the PDF and try to highlight text with your mouse. If text highlights, it works. If not, it is a scanned image and will not be readable.

---

## 📊 Metrics & Evaluation

FinSight tracks both **operational** and **RAG quality** metrics in real time via the sidebar dashboard.

### Operational Metrics

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

## ⚠️ Common Issues

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside the activated venv |
| `GROQ_API_KEY not found` | Check `.env` file exists and has no spaces around `=` sign |
| Slow first response | Normal — HuggingFace model loads into memory on first query |
| `Port already in use` | Run `streamlit run app.py --server.port 8502` |
| PDF not loading | Check the PDF is text-based, not a scanned image |
| Answer mixing categories | Make sure PDFs are placed in the correct subfolder under `data/` |

---

## 👤 Author

**Bala Sai Kiran Reddy**
ML Engineer | Data Scientist | Stuttgart, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/bala-sai-kiran-reddy-telluri-055370250/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/BalaTelluri)

---

*Built as part of a GenAI portfolio to demonstrate production-grade RAG architecture for financial document intelligence.*
