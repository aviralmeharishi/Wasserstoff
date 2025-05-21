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
import json # JSON parsing ke liye
import shutil # Directory operations ke liye (jaise persist directory saaf karna)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# API Key Handling
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st # Streamlit ko yahan import karna zaroori hai secrets access karne ke liye
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            print("GOOGLE_API_KEY loaded from Streamlit secrets.")
        else:
            print("GOOGLE_API_KEY not found in Streamlit secrets.")
            # Agar environment variable bhi nahi hai aur Streamlit secrets mein bhi nahi, toh error raise hoga
            if not os.environ.get("GOOGLE_API_KEY"): # Double check env var in case st fails
                 raise ValueError("GOOGLE_API_KEY not found. Please set it in deployment secrets or as an environment variable.")
    except ImportError:
        print("Streamlit not imported, checking only environment variable for GOOGLE_API_KEY.")
        if not os.environ.get("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable.")
    except Exception as e:
        print(f"An error occurred during GOOGLE_API_KEY retrieval: {e}")
        if not os.environ.get("GOOGLE_API_KEY"): # Final check if all else fails
            raise ValueError("GOOGLE_API_KEY could not be confirmed. Ensure it is set.")


# Initialize LLM and Embeddings
llm = None
embeddings = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        print("LLM and Embeddings initialized successfully using GOOGLE_API_KEY.")
    except Exception as e:
        print(f"Error initializing LLM or Embeddings with GOOGLE_API_KEY: {e}")
        # This suggests a problem with the key itself or Google's service availability
else:
    print("Critical Error: GOOGLE_API_KEY is not available. LLM and Embeddings cannot be initialized.")
    # Applications relying on llm or embeddings will fail.

EXTRACTED_TEXT_DIR = "extracted_texts_streamlit" 
PERSIST_DIRECTORY = "chroma_db_streamlit_store" 

vector_store = None
qa_chain = None

def initialize_vector_store(force_recreate=False):
    global vector_store, qa_chain, llm, embeddings

    if not llm or not embeddings:
        print("LLM or Embeddings not initialized. Cannot initialize vector store.")
        raise ConnectionError("LLM / Embeddings Seva shuru nahi hui. GOOGLE_API_KEY ya network connection check karein.")

    # Agar persist directory hai aur force_recreate nahi hai, toh load karne ki koshish karo
    if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
        print(f"Pehle se maujood Vector Store ko {PERSIST_DIRECTORY} se load karne ki koshish...")
        try:
            vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            print("Pehle se maujood Vector Store load ho gaya.")
        except Exception as e: # ChromaDB load hone mein error (e.g., different anaconda-project.xml)
            print(f"Pehle se maujood Vector Store ({PERSIST_DIRECTORY}) load karne mein error: {e}. Naya banane ki koshish karenge.")
            force_recreate = True 
            if os.path.exists(PERSIST_DIRECTORY): # Kharab directory ko hatane ki koshish
                try:
                    shutil.rmtree(PERSIST_DIRECTORY)
                    print(f"Kharab persist directory ({PERSIST_DIRECTORY}) hata di gayi hai.")
                except Exception as rm_e:
                    print(f"Kharab persist directory ({PERSIST_DIRECTORY}) hatane mein error: {rm_e}")
                    # Agar hatane mein error, toh naya banana bhi mushkil.
                    raise IOError(f"Kharab persist directory ({PERSIST_DIRECTORY}) ko hata nahi paye. Manual check zaroori hai.") from rm_e

    # Agar force_recreate hai ya vector_store abhi bhi None hai (load nahi hua ya pehle nahi tha)
    if force_recreate or not vector_store:
        print(f"Naya Vector Store '{EXTRACTED_TEXT_DIR}' directory ke documents se banaya ja raha hai...")
        if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
            print(f"'{EXTRACTED_TEXT_DIR}' mein koi text files nahi mili. Naya Vector Store nahi banaya ja sakta.")
            # Streamlit app mein iski सूचना (notification) user ko deni chahiye.
            # Yahan se return karne par vector_store None rahega.
            return 

        loader = DirectoryLoader(EXTRACTED_TEXT_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=False, silent_errors=True)
        documents = loader.load()

        if not documents:
            print("Koi documents load nahi hue. Vector Store nahi banaya ja sakta.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            print("Documents ko split karne ke baad koi text nahi mila. Vector Store nahi banaya ja sakta.")
            return
            
        print(f"Gemini embeddings aur naya vector store banaya ja raha hai. Thoda waqt lagega...")
        try:
            vector_store = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
            vector_store.persist() # Naye DB ko save karo
            print("Naya Gemini vector store ban gaya aur save ho gaya.")
        except Exception as e:
            print(f"Naya vector store banane mein error: {e}")
            # Streamlit app mein error dikhana chahiye
            raise IOError(f"Naya Knowledge Base banane mein error: {e}") from e


    # QA Chain ko initialize/update karo
    if vector_store:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print("Gemini QA Chain initialize ho gaya hai.")
    else:
        qa_chain = None # Agar vector_store nahi bana toh qa_chain bhi None hoga
        print("Vector Store initialize nahi hua, isliye QA Chain bhi initialize nahi hua.")


def ask_llm(question: str):
    global qa_chain
    if not qa_chain:
        return {"answer": "Error: Knowledge Base initialize nahi hua hai. Kripya pehle sidebar se initialize karein.", "sources": []}

    try:
        response = qa_chain({"query": question})
        answer = response.get("result", "Koi jawaab nahi mila.")
        
        sources = []
        if "source_documents" in response:
            # Extract base filename without extension for sources
            sources = list(set([os.path.splitext(os.path.basename(doc.metadata.get("source", "Unknown")))[0] for doc in response["source_documents"]]))
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"Error during Gemini LLM query: {e}")
        return {"answer": f"Sawaal process karne mein error (Gemini): {e}", "sources": []}

def extract_themes_llm():
    global vector_store, llm
    if not vector_store :
        return {"error": "Knowledge Base initialize nahi hua hai themes ke liye."}
    if not llm:
        return {"error": "LLM initialize nahi hua hai themes ke liye."}

    all_docs_for_themes = vector_store.similarity_search(
        query="key topics, concepts, and main ideas discussed across the documents", 
        k=min(25, len(vector_store.get().get('ids', []))) 
    ) 
    
    if not all_docs_for_themes:
        return {"error": "Knowledge Base mein themes nikalne ke liye koi documents nahi mile."}

    context_text = "\n\n".join([doc.page_content for doc in all_docs_for_themes])
    
    prompt = f"""
    Analyze the following text compiled from multiple documents and identify up to 5-7 major common themes.
    For each theme, provide:
    1. A concise title for the theme (key: "theme_title").
    2. A brief one or two-sentence description of the theme (key: "theme_description").
    3. A list of original document filenames (e.g., ["doc1", "reportA"]) that primarily support or discuss this theme (key: "supporting_documents"). Base these filenames on the source metadata.

    Present the output strictly as a JSON list of objects. Each object in the list should represent a theme and have ONLY the keys "theme_title", "theme_description", and "supporting_documents".

    Context:
    {context_text[:28000]} 

    JSON Output:
    """ 
    
    try:
        response = llm.invoke(prompt)
        themes_content_str = response.content
        
        print(f"Raw LLM response for themes: {themes_content_str}") 

        if themes_content_str.strip().startswith("```json"):
            themes_content_str = themes_content_str.strip()[7:]
            if themes_content_str.strip().endswith("```"):
                themes_content_str = themes_content_str.strip()[:-3]
        
        parsed_themes = json.loads(themes_content_str.strip())

        if not isinstance(parsed_themes, list):
            raise ValueError("LLM ne JSON list return nahi ki.")
        
        # Further validation for each theme object can be added here
        for theme_item in parsed_themes:
            if not all(key in theme_item for key in ["theme_title", "theme_description", "supporting_documents"]):
                print(f"Warning: Theme item {theme_item.get('theme_title','Unknown Theme')} is missing some keys.")
            if "supporting_documents" in theme_item and not isinstance(theme_item["supporting_documents"], list):
                if isinstance(theme_item["supporting_documents"], str): # If LLM gave a string
                    theme_item["supporting_documents"] = [s.strip() for s in theme_item["supporting_documents"].split(',')]
                else: # If not a string or list, make it an empty list or handle error
                    theme_item["supporting_documents"] = []


        return {"themes_data": parsed_themes}

    except json.JSONDecodeError as je:
        print(f"JSONDecodeError LLM se themes parse karte waqt: {je}")
        print(f"LLM ka raw output tha: {themes_content_str}")
        return {"error": "AI se themes parse nahi kar paye. Raw output mila.", "raw_output": themes_content_str}
    except ValueError as ve:
        print(f"Theme structure mein ValueError: {ve}")
        return {"error": f"AI ne themes anumaanit structure mein nahi diye: {ve}", "raw_output": themes_content_str}
    except Exception as e:
        print(f"Gemini theme extraction mein error: {e}")
        return {"error": f"Themes extract karne mein error (Gemini): {e}", "raw_output": "Error aaya, koi raw output nahi."}
