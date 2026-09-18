# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) based research assistant for analyzing and comparing a collection of research papers from the same research area.

The system allows users to ask questions about the loaded papers and provides several specialized research functions, including experimental result extraction, paper comparison, mini-survey generation, research gap identification, and limitation comparison.

The application is implemented as an interactive **Streamlit** web application.

---

## Overview

This project implements a RAG pipeline that combines:

* PDF document processing
* Text and table extraction
* Document chunking
* Vector embeddings
* ChromaDB vector storage
* Semantic retrieval
* Large Language Model (LLM) generation
* Function calling
* Interactive Streamlit interface

The system works with **four research papers** placed in the `articles/` directory. The papers are automatically discovered, processed, indexed, and made available to the research assistant.

---

## Main Features

### 1. Question Answering

Users can ask questions about the loaded papers through the chat interface.

The RAG pipeline retrieves the most relevant content from the papers and provides an answer based on the retrieved context.

The retrieval process searches each paper independently and returns the most relevant chunks based on embedding similarity.

If the answer cannot be found in the loaded papers, the system explicitly indicates that the information was not found in the provided documents and then uses the language model's general knowledge to provide an answer.

---

### 2. Experimental Results Extraction

The system can automatically extract the main experimental results reported in each paper.

The extracted information may include:

* Accuracy
* Precision
* Recall
* F1 Score
* mAP
* IoU
* Error Rate
* FLOPs
* Number of Parameters
* Dataset information

The system focuses on the primary benchmark dataset of each paper and converts reported error rates into accuracy when necessary.

---

### 3. Results Comparison

The extracted experimental results can be compared across the four papers.

The system generates:

* A numerical comparison table
* Bar charts for available numerical metrics

This allows users to compare reported performance metrics across the papers.

---

### 4. Mini Survey Generation

The system can automatically generate a short mini survey covering all four papers.

The generated survey includes:

* Problem introduction
* Per-paper method summaries
* Numerical comparison table
* Qualitative comparison of methods
* Progression of methods
* Conclusion

The survey is generated using information retrieved from the indexed papers and the extracted experimental results.

---

### 5. Research Gap Analysis

The system analyzes the conclusion, limitations, future work, and open-problem sections of the papers.

For each paper, it extracts:

* **Limitations**
* **Unsolved Problems**
* **Future Work**

---

### 6. Limitation Comparison

The limitations of the four papers can also be compared side by side.

The system provides:

* A comparison table of acknowledged limitations
* A qualitative discussion of shared and unique limitations

Only limitations explicitly identified in the papers are considered.

---

## RAG Pipeline

The main workflow of the system is:

```text
Research Papers (PDF)
        │
        ▼
   PDF Extraction
        │
        ▼
 Text & Table Processing
        │
        ▼
      Chunking
        │
        ▼
   Text Embeddings
        │
        ▼
     ChromaDB
   Vector Database
        │
        ▼
 Semantic Retrieval
        │
        ▼
 Relevant Context
        │
        ▼
      LLM
        │
        ▼
     Answer
```

The project uses `pymupdf4llm` to convert PDF papers into Markdown text. Tables are processed separately from prose, and the resulting content is divided into chunks before being embedded and stored in ChromaDB.

## Document Processing

### PDF Extraction

Research papers are loaded from the `articles/` directory.

```python
def load_pdf(path):
    return pymupdf4llm.to_markdown(path)
```

This converts the PDF content into Markdown-formatted text for further processing.

### Table Extraction

Tables are separated from the normal prose content before chunking.

This helps preserve the structure of experimental result tables and improves retrieval of numerical information.

### Chunking

The remaining prose is divided into overlapping chunks before being stored in the vector database.

The project uses a chunk size of 500 words with an overlap of 100 words during indexing.

---

## Vector Database

The project uses **ChromaDB** as its persistent vector database.

```python
db = chromadb.PersistentClient(path="./chroma_store")
collection = db.get_or_create_collection("rag_papers")
```

Each chunk is stored together with metadata identifying:

* The paper
* The chunk type (`table` or `prose`)

The generated embeddings are stored alongside the document chunks in the collection.

## Embeddings

The project uses:

```text
text-embedding-3-small
```

for generating vector representations of paper chunks and user queries.

During retrieval, the query is embedded and compared against the stored vectors in ChromaDB.

The system retrieves the top relevant chunks from each paper and sorts the results according to their similarity scores.

---

## Language Model

The project uses:

```text
gpt-4o-mini
```

as the chat model for generating answers and performing the higher-level research tasks.

The LLM is used for:

* Question answering
* Experimental result extraction
* Qualitative comparison
* Mini-survey generation
* Research gap extraction
* Limitation comparison

---

## Tool Calling

The research assistant provides five specialized functions:

```text
extract_experimental_results
compare_results
generate_survey
find_research_gap
compare_limitations
```

The LLM determines whether a user request matches one of these functions and calls the appropriate function when necessary. Otherwise, the request is handled through the standard RAG question-answering pipeline.

---

## Quick Actions

The Streamlit interface provides several quick actions:

* 📊 Extract experimental results
* 📈 Compare results
* 📝 Generate mini survey
* 🔍 Find research gaps
* ⚖️ Compare limitations

These actions allow common research-analysis tasks to be executed directly without manually writing a query.

---

## Streamlit Interface

The application provides an interactive web interface built with Streamlit.

The interface includes:

* Loaded paper list
* Indexing log
* Quick actions
* Chat interface
* Clear chat functionality

The application initializes the knowledge base on the first run and caches the backend using Streamlit's `cache_resource`.

---

## Project Structure

```text
rag-research-assistant/
│
├── articles/
│   ├── paper_1.pdf
│   ├── paper_2.pdf
│   ├── paper_3.pdf
│   └── paper_4.pdf
│
├── chroma_store/
│
├── app.py
├── backend.py
├── main.ipynb
└── README.md
```

The `articles/` directory should contain the four research papers used by the system.

The `chroma_store/` directory is used for persistent storage of the ChromaDB vector collection.

---

## Environment Variables

The application supports the following environment variables:

```text
RESEARCH_ASSISTANT_API_KEY
RESEARCH_ASSISTANT_BASE_URL
RESEARCH_ASSISTANT_ARTICLES_DIR
RESEARCH_ASSISTANT_CHROMA_DIR
```

Example:

```bash
RESEARCH_ASSISTANT_API_KEY="your-api-key"
RESEARCH_ASSISTANT_BASE_URL="https://api.gapgpt.app/v1"
RESEARCH_ASSISTANT_ARTICLES_DIR="articles"
RESEARCH_ASSISTANT_CHROMA_DIR="./chroma_store"
```

The API key and base URL are used to initialize the OpenAI-compatible client.

---

**## API Key Requirement**

Before installing and running the application, users must provide their own API key. The API key is required for communication with the OpenAI-compatible API and for the application's language model functionality.

Set your API key in the `RESEARCH_ASSISTANT_API_KEY` environment variable before running the application:

```bash
RESEARCH_ASSISTANT_API_KEY="your-api-key"
```

Replace `"your-api-key"` with your personal API key.

---

## Installation

Install the required Python packages:

```bash
pip install streamlit
pip install openai
pip install chromadb
pip install pymupdf
pip install pymupdf4llm
pip install pandas
pip install matplotlib
```

Then place the four research papers inside:

```text
articles/
```

---

## Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

On the first run, the system:

1. Discovers the PDF papers.
2. Extracts their content.
3. Separates tables from prose.
4. Creates document chunks.
5. Generates embeddings.
6. Builds the ChromaDB collection.
7. Makes the papers available for retrieval and analysis.

If the existing index already matches the current set of papers, the system skips rebuilding it.

---

## Example Queries

Users can ask questions such as:

```text
What problem does each paper address?

What is the main contribution of each paper?

Compare the experimental results of all four papers.

Which datasets are used in the papers?

What are the limitations mentioned by each paper?

What future work is suggested?

What research gaps are identified across these papers?

Generate a mini survey covering all four papers.
```

---

## Technologies

* Python
* Streamlit
* OpenAI-compatible API
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* ChromaDB
* PyMuPDF
* PyMuPDF4LLM
* Pandas
* Matplotlib

---

## Key Concepts

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Text Embeddings
* Document Chunking
* PDF Processing
* LLMs
* Function Calling
* Research Paper Analysis
* Experimental Result Extraction
* Research Gap Analysis
* Comparative Analysis
* Mini Survey Generation
* Information Retrieval

---

## Course

**Data Mining Course — University of Isfahan**

This project was developed as a course assignment for the **Data Mining** course at the **University of Isfahan**, with the aim of gaining practical experience with **Retrieval-Augmented Generation (RAG)** and **Function Calling** using Large Language Models (LLMs).