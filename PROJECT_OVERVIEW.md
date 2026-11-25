# Project overview for college documentation

## Problem statement
Students and general users often encounter cybersecurity information across many lengthy publications and may find it difficult to locate reliable explanations quickly. CyberSage AI provides a question-answering interface over a curated collection of cybersecurity PDFs.

## Objectives
- Convert trusted PDF references into a searchable knowledge base.
- Use semantic and keyword retrieval to locate relevant passages.
- Generate student-friendly answers grounded in retrieved material.
- Display the document and page used for the answer.
- Demonstrate a complete RAG workflow with a vector database, embedding model, language model API, API backend and web interface.

## Scope
The system covers defensive security awareness and learning, including phishing prevention, password hygiene, device security, network safety, web security concepts and incident-response awareness. It does not perform live system scans or offensive security operations.

## Method
PDF text is extracted with PyPDF, split into overlapping chunks by LangChain, embedded with FastEmbed, and stored in Qdrant. At query time, Qdrant retrieves semantically related passages while Rank-BM25 retrieves keyword matches. Reciprocal Rank Fusion merges both result lists. LangGraph coordinates retrieval and answer generation, while NVIDIA Nemotron produces an answer from the selected context. The answer includes source and page references.

## Evaluation plan
Prepare a held-out set of questions with expected source documents and pages. Evaluate whether relevant passages appear in the top-k retrieved results, whether responses cite the supporting source, and whether answers correctly state when the indexed documents do not contain enough information. Record failed cases and revise the document collection, chunking and retrieval settings.
