import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Fallback for local testing if GOOGLE_API_KEY is not in Streamlit Cloud secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        else:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in Streamlit Cloud secrets or as an environment variable.")
    except ImportError:
         raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable.")
    except Exception as e:
        raise e


# Initialize LLM and Embeddings (ensure API key is available)
llm = None
embeddings = None
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not configured. LLM and Embeddings will not be initialized.")
    # Optionally raise an error or use a dummy/mock object for local testing without API key
    # For now, we'll let it proceed, but functions using llm/embeddings will fail if not initialized.


EXTRACTED_TEXT_DIR = "extracted_texts_streamlit" # Should match the one in app_streamlit.py
PERSIST_DIRECTORY = "chroma_db_streamlit_store" # Using a different name for DB

vector_store = None
qa_chain = None

def initialize_vector_store(force_recreate=False):
    global vector_store, qa_chain, llm, embeddings

    if not llm or not embeddings:
        print("LLM or Embeddings not initialized due to missing GOOGLE_API_KEY. Cannot initialize vector store.")
        raise ConnectionError("LLM or Embeddings not initialized. Check GOOGLE_API_KEY.") # Or handle gracefully

    if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
        print(f"Loading existing Gemini vector store from {PERSIST_DIRECTORY}")
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        print(f"Creating new Gemini vector store from documents in {EXTRACTED_TEXT_DIR}")
        if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
            print(f"No text files found in {EXTRACTED_TEXT_DIR}. Cannot create vector store.")
            # In Streamlit, you might want to inform the user via st.warning
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
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        vector_store.persist()
        print("Gemini vector store created and persisted.")

    if vector_store:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    else: # Ensure qa_chain is None if vector_store failed to initialize
        qa_chain = None 
    print("Gemini QA Chain Initialized (or re-checked).")


def ask_llm(question: str):
    global qa_chain
    if not qa_chain: # This check implies vector_store should also be initialized
        print("QA chain not initialized. Please ensure knowledge base is initialized.")
        # In Streamlit, you would show this error on the UI.
        # For this backend function, raising an error or returning a specific error message is better.
        # initialize_vector_store() # Avoid re-initializing here without user action in UI
        return {"answer": "Error: Knowledge Base not initialized. Please initialize it first via the sidebar.", "sources": []}

    try:
        response = qa_chain({"query": question})
        answer = response.get("result", "No answer found.")
        
        sources = []
        if "source_documents" in response:
            sources = list(set([doc.metadata.get("source", "Unknown").split('/')[-1].replace(".txt","") for doc in response["source_documents"]]))
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"Error during Gemini LLM query: {e}")
        return {"answer": f"Error processing question with Gemini: {e}", "sources": []}

def extract_themes_llm():
    global vector_store, llm # Ensure llm is accessible
    if not vector_store:
        print("Vector store not initialized. Cannot extract themes.")
        return {"themes_text": "Error: Knowledge Base not initialized for theme extraction.", "supporting_documents": []}
    if not llm:
        print("LLM not initialized. Cannot extract themes.")
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
    # Increased context for Gemini 1.5 Flash, ensure it's within model limits
    
    try:
        response = llm.invoke(prompt)
        themes_content = response.content
        
        # Attempt to extract source filenames mentioned by the LLM if it follows the prompt
        # This is a simple heuristic. Better approach would be more complex.
        theme_sources = list(set([doc.metadata.get("source", "Unknown").split('/')[-1].replace(".txt","") for doc in all_docs_for_themes if doc.metadata.get("source", "Unknown").split('/')[-1].replace(".txt","") in themes_content]))
        if not theme_sources and all_docs_for_themes : # Fallback if LLM doesn't list sources but context was used
             theme_sources = ["Based on processed documents like: " + ", ".join(list(set(doc.metadata.get("source", "Unknown").split('/')[-1].replace(".txt","") for doc in all_docs_for_themes[:3])))]


        return {"themes_text": themes_content, "supporting_documents": theme_sources }
    except Exception as e:
        print(f"Error during Gemini theme extraction: {e}")
        return {"themes_text": f"Error extracting themes with Gemini: {e}", "supporting_documents": []}
