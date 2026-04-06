from __future__ import annotations
from typing import List, Dict, Tuple
import os
import time
import numpy as np

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

SYSTEM_PROMPT = """You are FinSight, a highly accurate financial document assistant.

STRICT RULES - follow exactly:
1. Answer ONLY using the context provided below - never use outside knowledge
2. ALWAYS write your answer as a clear, well-structured paragraph - never use bullet points unless the user specifically asks for a list
3. Be precise - include exact numbers, limits, percentages, dates when available
4. If the context contains a partial answer - write a paragraph with only what is confirmed
5. If the user asks to summarize or write as a paragraph - always do so using whatever information is available in the context
6. If the context does NOT contain ANY relevant information at all - respond with exactly: NO_ANSWER
7. Never guess, assume or infer beyond what is written in the context
8. Never repeat the question back in your answer
9. Write in plain, professional English that is easy to understand"""


def _expand_query(client: Groq, question: str) -> str:
    """Rewrite the question to improve retrieval accuracy."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Rewrite the following question to be more detailed and specific for searching financial documents. Add relevant financial terms. Return ONLY the rewritten question, nothing else."},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return question


def _format_docs(docs: List[Document]) -> str:
    filtered = [d for d in docs if len(d.page_content.strip()) > 30]
    return "\n\n".join(d.page_content for d in filtered)[:3000]


def _compute_confidence(scores: list) -> float:
    """
    Convert FAISS L2 distances to confidence percentage.
    Lower L2 distance = better match.
    Typical good scores: 0.0-0.5 = excellent, 0.5-1.0 = good, 1.0+ = poor
    """
    if not scores:
        return 0.0
    best_score = min(scores)
    # Sigmoid-based conversion for better range
    # score 0.0 -> 100%, score 0.5 -> 85%, score 1.0 -> 60%, score 2.0 -> 20%
    confidence = 100.0 * (1.0 / (1.0 + best_score))
    return round(min(max(confidence, 0.0), 100.0), 1)


def build_category_chains(category_docs: Dict[str, List[Document]]):
    # Better embedding model for higher accuracy
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    retrievers = {}
    for category, docs in category_docs.items():
        vectorstore = FAISS.from_documents(docs, embeddings)
        retrievers[category] = (
            vectorstore,
            vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        )
        print(f"Loaded: {category}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    client = Groq(api_key=api_key)
    return retrievers, client


def query_rag(retrievers: dict, client: Groq, question: str, category: str):
    if category not in retrievers:
        return "No documents loaded for this category.", [], False, 0.0, 0.0, ""

    vectorstore, retriever = retrievers[category]
    start_time = time.time()

    # Step 1: Expand query for better retrieval
    expanded_query = _expand_query(client, question)

    # Step 2: Search with expanded query
    docs_with_scores = vectorstore.similarity_search_with_score(expanded_query, k=5)

    # Step 3: Also search with original query and merge results
    original_docs = vectorstore.similarity_search_with_score(question, k=3)

    # Merge and deduplicate by content
    seen = set()
    merged = []
    for doc, score in docs_with_scores + original_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append((doc, score))

    # Sort by score ascending (lower = better)
    merged.sort(key=lambda x: x[1])
    merged = merged[:5]

    docs   = [d for d, _ in merged]
    scores = [s for _, s in merged]

    confidence = _compute_confidence(scores)

    context = _format_docs(docs)
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    # Step 4: Ask Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.0,
        max_tokens=600
    )
    answer = response.choices[0].message.content.strip()

    response_time = round(time.time() - start_time, 2)
    found = answer.strip().upper() != "NO_ANSWER" and len(answer) > 15

    return answer.strip(), sources, found, response_time, confidence, context


def _format_docs(docs: List[Document]) -> str:
    filtered = [d for d in docs if len(d.page_content.strip()) > 30]
    # Rerank — put highest scored chunks first
    return "\n\n".join(d.page_content for d in filtered)[:3000]


def build_category_chains(category_docs: Dict[str, List[Document]]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    retrievers = {}
    for category, docs in category_docs.items():
        vectorstore = FAISS.from_documents(docs, embeddings)
        retrievers[category] = (
            vectorstore,
            vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        )
        print(f"✓ FAISS index built for: {category}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    client = Groq(api_key=api_key)
    return retrievers, client


def query_rag(retrievers: dict, client: Groq, question: str, category: str):
    """
    Returns (answer, sources, found, response_time, confidence, context)
    """
    if category not in retrievers:
        return "No documents loaded for this category.", [], False, 0.0, 0.0, ""

    vectorstore, retriever = retrievers[category]

    start_time = time.time()

    # Retrieve + confidence score
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=4)

    # Sort by score ascending (lower L2 distance = more relevant)
    docs_with_scores.sort(key=lambda x: x[1])

    docs   = [d for d, _ in docs_with_scores]
    scores = [s for _, s in docs_with_scores]

    if scores:
        avg_distance = sum(scores) / len(scores)
        confidence   = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))
    else:
        confidence = 0.0

    # High confidence threshold — only answer if retrieval is confident
    CONFIDENCE_THRESHOLD = 0.25
    if confidence < CONFIDENCE_THRESHOLD:
        return "NO_ANSWER", [], False, round(time.time() - start_time, 2), round(confidence * 100, 1), ""

    context = _format_docs(docs)
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    # Ask Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.0,
        max_tokens=600
    )
    answer = response.choices[0].message.content.strip()

    response_time = round(time.time() - start_time, 2)
    found = answer.strip().upper() != "NO_ANSWER" and len(answer) > 15

    return answer.strip(), sources, found, response_time, round(confidence * 100, 1), context
