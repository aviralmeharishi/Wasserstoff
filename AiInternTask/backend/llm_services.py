# llm_services.py

# --- SQLITE3 VERSION FIX (pysqlite3-binary) ---
# Yeh code baaki saare imports se PEHLE aana chahiye
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3-binary for SQLite")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3 (might cause issues with ChromaDB)")
    pass
# --- END OF SQLITE3 VERSION FIX ---

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# API Key Handling
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        else:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in Streamlit Cloud secrets or as an environment variable.")
    except ImportError:
         raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable.")
    except Exception as e: # Catch any other exception during streamlit import or secrets access
        print(f"Error accessing Streamlit secrets for GOOGLE_API_KEY: {e}")
        # If Streamlit is not available or secrets don't have the key, this path will be taken.
        # The script should ideally halt or clearly indicate that critical functionality will be missing.
        # For now, will rely on the llm/embeddings initialization check below.
        pass


# Initialize LLM and Embeddings
llm = None
embeddings = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        print("LLM and Embeddings initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM or Embeddings with GOOGLE_API_KEY: {e}")
        # This indicates a problem with the API key or the Google service itself.
else:
    print("Warning: GOOGLE_API_KEY not configured. LLM and Embeddings will not be initialized.")


EXTRACTED_TEXT_DIR = "extracted_texts_streamlit" 
PERSIST_DIRECTORY = "chroma_db_streamlit_store" 

vector_store = None
qa_chain = None

def initialize_vector_store(force_recreate=False):
    global vector_store, qa_chain, llm, embeddings

    if not llm or not embeddings:
        print("LLM or Embeddings not initialized. Cannot initialize vector store.")
        # In Streamlit app, this should be shown as an error to the user.
        raise ConnectionError("LLM or Embeddings not initialized. Check GOOGLE_API_KEY or network.")

    # Check if extracted text directory exists and has files, only if we are creating a new store
    if force_recreate or not os.path.exists(PERSIST_DIRECTORY):
        print(f"Attempting to create new vector store from documents in {EXTRACTED_TEXT_DIR}")
        if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
            print(f"No text files found in {EXTRACTED_TEXT_DIR}. Cannot create new vector store.")
            # Inform Streamlit user if possible
            if 'st' in globals() and hasattr(st, 'warning'): # Check if streamlit is imported and usable
                st.warning(f"No processed text files found in '{EXTRACTED_TEXT_DIR}' to build knowledge base.")
            return # Stop if no files to process for a new DB

    if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
        print(f"Loading existing Gemini vector store from {PERSIST_DIRECTORY}")
        try:
            vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            print("Existing vector store loaded.")
        except Exception as e:
            print(f"Error loading existing vector store from {PERSIST_DIRECTORY}: {e}. Will try to recreate.")
            # Fallback to recreating if loading fails
            force_recreate = True # Set force_recreate to true to attempt recreation
            # Clean up potentially corrupted persist directory before recreating
            if os.path.exists(PERSIST_DIRECTORY):
                try:
                    shutil.rmtree(PERSIST_DIRECTORY)
                    print(f"Removed corrupted persist directory: {PERSIST_DIRECTORY}")
                except Exception as rm_e:
                    print(f"Error removing corrupted persist directory {PERSIST_DIRECTORY}: {rm_e}")
                    # If removal fails, creating new one might also fail.
                    return


    if force_recreate or not vector_store: # If loading failed or was skipped
        print(f"Creating new Gemini vector store from documents in {EXTRACTED_TEXT_DIR}")
        # This block should ideally be protected by the check for EXTRACTED_TEXT_DIR having files
        if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
             print(f"Still no text files found in {EXTRACTED_TEXT_DIR} on recreate attempt. Cannot create vector store.")
             return

        loader = DirectoryLoader(EXTRACTED_TEXT_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=False, silent_errors=True)
        documents = loader.load()

        if not documents:
            print("No documents loaded. Cannot create vector store.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            print("No texts available after splitting. Cannot create vector store.")
            return
            
        print(f"Creating Gemini embeddings and new vector store. This may take a moment...")
        try:
            vector_store = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
            vector_store.persist()
            print("Gemini vector store created and persisted.")
        except Exception as e:
            print(f"Error creating new vector store: {e}")
            # Inform Streamlit user if possible
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"Error building knowledge base: {e}")
            return # Stop if DB creation fails

    if vector_store:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    else: 
        qa_chain = None 
    print("Gemini QA Chain Initialized (or re-checked).")


def ask_llm(question: str):
    global qa_chain
    if not qa_chain:
        return {"answer": "Error: Knowledge Base not initialized. Please initialize it first via the sidebar.", "sources": []}

    try:
        response = qa_chain({"query": question})
        answer = response.get("result", "No answer found.")
        
        sources = []
        if "source_documents" in response:
            sources = list(set([os.path.splitext(os.path.basename(doc.metadata.get("source", "Unknown")))[0] for doc in response["source_documents"]]))
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"Error during Gemini LLM query: {e}")
        return {"answer": f"Error processing question with Gemini: {e}", "sources": []}

def extract_themes_llm():
    global vector_store, llm
    if not vector_store :
        return {"themes_text": "Error: Knowledge Base not initialized for theme extraction.", "supporting_documents": []}
    if not llm:
        return {"themes_text": "Error: LLM not initialized for theme extraction.", "supporting_documents": []}

    all_docs_for_themes = vector_store.similarity_search(query="key topics and concepts", k=min(20, len(vector_store.get().get('ids', [])))) 
    
    if not all_docs_for_themes:
        return {"themes_text": "No documents found to extract themes from.", "supporting_documents": []}

    context_text = "\n\n".join([doc.page_content for doc in all_docs_for_themes])
    
    prompt = f"""
    Analyze the following text from multiple documents and identify up to 5 major common themes.
    For each theme, briefly describe it and list the original document filenames (e.g., 'doc1', 'reportA') that seem to support or discuss this theme.
    Present the output clearly.

    Context:
    {context_text[:25000]}

    Identified Themes:
    """ 
    
    try:
        response = llm.invoke(prompt)
        themes_content = response.content
        
        theme_sources_candidates = [os.path.splitext(os.path.basename(doc.metadata.get("source", "Unknown")))[0] for doc in all_docs_for_themes]
        theme_sources = list(set([s for s in theme_sources_candidates if s in themes_content])) # Simple check
        
        if not theme_sources and all_docs_for_themes:
             theme_sources = ["Based on processed documents like: " + ", ".join(list(set(theme_sources_candidates[:3])))]

        return {"themes_text": themes_content, "supporting_documents": theme_sources }
    except Exception as e:
        print(f"Error during Gemini theme extraction: {e}")
        return {"themes_text": f"Error extracting themes with Gemini: {e}", "supporting_documents": []}
