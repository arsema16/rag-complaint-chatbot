# 🏦 CrediTrust Financial - RAG-Powered Complaint Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Gradio-4.16+-orange.svg)](https://gradio.app/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-red.svg)](https://github.com/facebookresearch/faiss)

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Solution Architecture](#solution-architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Tasks Completed](#tasks-completed)
  - [Task 1: EDA and Preprocessing](#task-1-eda-and-preprocessing)
  - [Task 2: Chunking and Vector Store](#task-2-chunking-and-vector-store)
  - [Task 3: RAG Pipeline](#task-3-rag-pipeline)
  - [Task 4: Interactive UI](#task-4-interactive-ui)
- [Evaluation Results](#evaluation-results)
- [Usage Guide](#usage-guide)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

---

## 🎯 Overview

This project builds an **Intelligent Complaint Analysis System** for CrediTrust Financial, a fast-growing digital finance company serving East African markets. The system uses **Retrieval-Augmented Generation (RAG)** to transform raw, unstructured customer complaints into actionable business insights.

**Key Features:**
- 📊 Processes complaints across 4 product lines
- 🔍 Semantic search using FAISS vector database
- 🤖 RAG-powered question answering
- 🖥️ Interactive Gradio UI for non-technical users
- 📈 Evidence-backed answers with source attribution

---

## 💼 Business Problem

CrediTrust Financial receives thousands of customer complaints monthly across:
- **Credit Cards**
- **Personal Loans**
- **Savings Accounts**
- **Money Transfers**

**Current Challenges:**
- ❌ Customer Support overwhelmed by complaint volume
- ❌ Product Managers spend hours manually reading complaints
- ❌ Compliance & Risk teams react slowly to issues
- ❌ Executives lack visibility into emerging pain points

**Our Solution:**
- ✅ Reduce trend identification time from **days to minutes**
- ✅ Empower non-technical teams with self-service insights
- ✅ Shift from reactive to proactive issue identification

---

## 🏗️ Solution Architecture
┌─────────────────────────────────────────────────────────────────┐
│ DATA PIPELINE │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Load Data │───▶│ Preprocess │───▶│ Chunk │ │
│ │ (50K rows) │ │ (Cleaning) │ │ (500 char) │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Embeddings │───▶│ Vector DB │───▶│ FAISS Index │ │
│ │ (384-dim) │ │ (ChromaDB) │ │ (400 chunks)│ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ RAG PIPELINE │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Query │───▶│ Retrieve │───▶│ Generate │ │
│ │ Question │ │ (Top-k) │ │ Answer │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Embed Query │ │ FAISS │ │ Context │ │
│ │ (384-dim) │ │ Search │ │ + Sources │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ UI LAYER │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ GRADIO INTERFACE │ │
│ │ ┌────────────────────────────────────────────────────┐ │ │
│ │ │ Question: "Why are customers unhappy with...?" │ │ │
│ │ └────────────────────────────────────────────────────┘ │ │
│ │ ┌────────────────────────────────────────────────────┐ │ │
│ │ │ Answer: "Customers complain about hidden fees..." │ │ │
│ │ └────────────────────────────────────────────────────┘ │ │
│ │ ┌────────────────────────────────────────────────────┐ │ │
│ │ │ Sources: │ │ │
│ │ │ 1. "The credit card fees increased..." │ │ │
│ │ │ 2. "I was charged unexpected annual fees..." │ │ │
│ │ └────────────────────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘

---

## 📁 Project Structure
```
rag-complaint-chatbot/
├── .github/
│ └── workflows/
│ └── unittests.yml # CI/CD pipeline
├── .vscode/
│ └── settings.json # VS Code config
├── data/
│ ├── raw/
│ │ └── complaints.csv # Original CFPB data
│ ├── processed/
│ │ ├── product_distribution.png
│ │ ├── narrative_analysis.png
│ │ └── top_issues_by_product.png
│ ├── filtered_complaints.csv # Cleaned data (Task 1)
│ └── chunked_complaints.csv # Chunked data (Task 2)
├── notebooks/
│ ├── eda_preprocessing.ipynb # EDA notebook
│ └── README.md
├── src/
│ ├── init.py
│ ├── data_loader.py # Data loading module
│ ├── run_all.py # Complete pipeline (Tasks 1-2)
│ ├── chunking.py # Chunking module
│ ├── vector_store.py # Vector store module
│ ├── rag_pipeline.py # RAG pipeline (Task 3)
│ └── evaluation.py # Evaluation module
├── vector_store/
│ └── faiss/
│ ├── index.faiss # FAISS vector index
│ └── metadata.pkl # Chunk metadata
├── tests/
│ ├── init.py
│ ├── test_chunking.py
│ └── test_vector_store.py
├── app.py # Gradio UI (Task 4)
├── download_data.py # Data download script
├── requirements.txt # Dependencies
├── INTERIM_REPORT.md # Interim submission report
├── README.md # This file
├── .gitignore
└── LICENSE
```

---

## 🚀 Setup and Installation

### Prerequisites

- Python 3.11+
- Git
- Virtual environment (venv)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the CFPB dataset
python download_data.py

# 5. Run the complete pipeline (Tasks 1-2)
python src/run_all.py

# 6. Test RAG pipeline (Task 3)
python src/rag_pipeline.py

# 7. Launch the UI (Task 4)
python app.py
