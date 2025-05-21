# AI Document Chatbot & Theme Identifier

**Project developed for the Wasserstoff Gen-AI Internship Task.**

This application leverages Generative AI to provide an interactive platform for document analysis. Users can upload multiple documents, receive answers to queries based on their content with precise citations, and automatically extract common themes present across the document set.

**🔗 Live Demo URL:** [https://wasserstoff-ai-chatbot.streamlit.app/](https://wasserstoff-ai-chatbot.streamlit.app/)

---

## 📜 Overview

The primary objective of this project is to create an intelligent system capable of ingesting a large corpus of documents (minimum 75, supporting formats like PDF, scanned images, and TXT). The system allows users to perform research by asking natural language questions and to identify overarching themes synthesized from the processed documents. This project emphasizes both research-driven development and practical implementation of Generative AI applications.

---

## ✨ Key Features

* **Multi-Format Document Upload:** Supports uploading various document types including PDF, PNG, JPG, JPEG, and TXT files.
* **OCR for Scanned Documents:** Integrated Optical Character Recognition to convert scanned images and image-based PDFs into machine-readable text.
* **Interactive Question Answering:** Users can ask questions in natural language and receive answers derived from the uploaded documents, complete with source document citations.
* **Automated Theme Identification:** The application analyzes the content across all processed documents to identify and present common themes in a structured, tabular format.
* **User-Friendly Interface:** Built with Streamlit for a clean, intuitive, and interactive user experience.

---

## 🛠️ Technology Stack

* **Application Framework:** Streamlit
* **Core Language:** Python
* **LLM Orchestration:** LangChain
* **Generative AI Model:** Google Gemini API
* **Vector Database:** ChromaDB
* **Text Extraction & OCR:** PdfPlumber, Pytesseract, Pillow
* **Data Handling:** Pandas

---

## 🚀 How to Use the Deployed Application

1.  **Upload Documents:** Navigate to the "Upload & Process Documents" section. Select your files (PDFs, images, TXT) using the file uploader.
2.  **Process Files:** After selecting files, click the "Process Uploaded Files" button. This will save the files and extract their text content.
3.  **Initialize Knowledge Base:** Once files are processed, go to the sidebar and click the "Initialize/Re-Initialize Knowledge Base" button. This step prepares the documents for querying and theme extraction and may take some time as models are loaded and the database is built.
4.  **Ask Questions:** After the Knowledge Base is successfully initialized (indicated in the sidebar), go to the "Ask Questions" section. Type your question and click "Ask Question" to receive an answer and cited sources.
5.  **Extract Themes:** Similarly, after Knowledge Base initialization, go to the "Extract Themes" section and click "Extract Common Themes" to view a table of identified themes and their supporting documents.

---

## 📁 Project Structure Overview
.
├── app_streamlit.py         # Main Streamlit application script
├── requirements.txt         # Python package dependencies
├── packages.txt             # System-level dependencies for deployment (e.g., Tesseract)
├── backend/
│   ├── __init__.py          # Makes 'backend' a Python package (must be this exact name)
│   ├── llm_services.py      # Handles LLM interactions, vector store, Q&A, and theme logic
│   └── mere_functions.py    # Contains functions for text extraction and OCR
├── data_streamlit_uploads/    # (Auto-created by app) Stores raw uploaded files temporarily
├── extracted_texts_streamlit/ # (Auto-created by app) Stores text extracted from documents
└── chroma_db_streamlit_store/ # (Auto-created by app) Persistent storage for the Vector Database

---

This project aims to demonstrate practical skills in Generative AI, machine learning lifecycle management, API integration, and data-driven application development.

