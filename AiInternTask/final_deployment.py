import streamlit as st
import os
import shutil
import pandas as pd # For DataFrame
import textwrap     # For wrapping text in table cells

# Import your backend functions
from backend.mere_functions import process_all_documents
from backend.llm_services import initialize_vector_store, ask_llm, extract_themes_llm

# --- Page Config & Title ---
st.set_page_config(page_title="AI Document Chatbot & Theme Identifier", layout="wide")
st.title("📄 AI Document Chatbot & Theme Identifier")
st.caption("Internship Task Submission")

# --- Directory Setup ---
DATA_DIR = "data_streamlit_uploads"
EXTRACTED_TEXT_DIR = "extracted_texts_streamlit"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

# --- Session State for Knowledge Base Initialization ---
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Actions")
    if st.button("Initialize/Re-Initialize Knowledge Base", key="init_kb_sidebar"):
        with st.spinner("Initializing Knowledge Base... (Models are loading, this may take some time)"):
            try:
                if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
                    st.warning(f"No processed text files found in '{EXTRACTED_TEXT_DIR}'. Please upload and process files first.")
                    st.session_state.vector_store_initialized = False 
                else:
                    initialize_vector_store(force_recreate=True)
                    st.session_state.vector_store_initialized = True
                    st.success("Knowledge Base Initialized!")
            except Exception as e:
                st.error(f"Error initializing Knowledge Base: {e}")
                st.session_state.vector_store_initialized = False
    
    if st.session_state.vector_store_initialized:
        st.success("Knowledge Base is ready!")
    else:
        st.info("Knowledge Base is not yet initialized. Please process files and then initialize.")

    st.markdown("---")
    st.header("📖 Instructions")
    st.markdown("""
    1.  **Upload Documents:** Select files and click the "Process Uploaded Files" button.
    2.  **Initialize Knowledge Base:** After files are processed, click the "Initialize/Re-Initialize Knowledge Base" button in this sidebar. This will take some time.
    3.  **Ask Questions / Extract Themes:** Once the Knowledge Base is initialized, you can ask questions or extract themes on the main page.
    """)

# --- Main Page Content ---
# --- Step 1: File Upload & Processing ---
st.header("1. Upload & Process Documents")
uploaded_files = st.file_uploader("Select PDF, PNG, JPG, JPEG, or TXT files", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    # Clear previous uploads and extracted texts to process only current batch
    for directory_to_clear in [DATA_DIR, EXTRACTED_TEXT_DIR]:
        if os.path.exists(directory_to_clear):
            for f_name_to_delete in os.listdir(directory_to_clear):
                try:
                    os.remove(os.path.join(directory_to_clear, f_name_to_delete))
                except Exception as e_del:
                    print(f"Could not delete old file {f_name_to_delete} from {directory_to_clear}: {e_del}")
    
    st.info(f"{len(uploaded_files)} file(s) selected. Saving files...")
    
    for uploaded_file_item in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file_item.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file_item.getbuffer())
        except Exception as e:
            st.error(f"Error saving '{uploaded_file_item.name}': {e}")
    st.success(f"All {len(uploaded_files)} selected files have been saved to '{DATA_DIR}'.")

    if st.button("Process Uploaded Files", key="process_button"):
        if not os.listdir(DATA_DIR):
            st.warning("No files found in the upload directory to process. Please upload files first.")
        else:
            with st.spinner(f"Processing {len(os.listdir(DATA_DIR))} file(s)... (Text extraction)"):
                try:
                    process_all_documents(DATA_DIR, EXTRACTED_TEXT_DIR)
                    st.success(f"All files processed! Extracted text saved in '{EXTRACTED_TEXT_DIR}'.")
                    st.info("You can now initialize/re-initialize the Knowledge Base (button in the sidebar).")
                    st.session_state.vector_store_initialized = False 
                except Exception as e:
                    st.error(f"Error processing files: {e}")

st.markdown("---")

# --- Step 2: Question Answering ---
st.header("2. Ask Questions")
if not st.session_state.vector_store_initialized:
    st.warning("Please initialize the Knowledge Base from the sidebar after processing files.")

question = st.text_input("Type your question here...", key="question_input_main", disabled=not st.session_state.vector_store_initialized)
if st.button("Ask Question", key="ask_button_main", disabled=not st.session_state.vector_store_initialized):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Finding an answer..."):
            try:
                response = ask_llm(question)
                st.markdown("**Answer:**")
                st.text_area("Answer", value=response.get("answer", "No answer found."), height=150, key="qna_answer_area", disabled=True)
                if response.get("sources"):
                    st.markdown("**Sources (files contributing to the answer):**")
                    st.write(", ".join(response.get("sources")))
            except Exception as e:
                st.error(f"Error processing question: {e}")

st.markdown("---")

# --- Step 3: Theme Extraction ---
st.header("3. Extract Themes")
if st.button("Extract Common Themes", key="themes_button_main", disabled=not st.session_state.vector_store_initialized):
    with st.spinner("Extracting themes... (The AI model may take some time)"):
        try:
            response_data = extract_themes_llm()
            
            if "error" in response_data:
                st.error(f"Error during theme extraction: {response_data['error']}")
                if "raw_output" in response_data:
                    st.info("Raw AI output (if available):")
                    st.text(response_data["raw_output"])
            elif "themes_data" in response_data and response_data["themes_data"]:
                themes_list = response_data["themes_data"]
                
                # Apply text wrapping to theme descriptions for better readability in table
                WRAP_WIDTH = 90 # Adjust as needed
                for theme_item in themes_list:
                    if "theme_description" in theme_item and isinstance(theme_item["theme_description"], str):
                        theme_item["theme_description"] = textwrap.fill(theme_item["theme_description"], width=WRAP_WIDTH)
                    # Optionally wrap title if it can be long
                    # if "theme_title" in theme_item and isinstance(theme_item["theme_title"], str):
                    #     theme_item["theme_title"] = textwrap.fill(theme_item["theme_title"], width=WRAP_WIDTH // 2)


                df_themes = pd.DataFrame(themes_list)
                
                cols_to_display = ["theme_title", "theme_description", "supporting_documents"]
                df_display = df_themes[[col for col in cols_to_display if col in df_themes.columns]].copy()
                
                st.markdown("**Identified Themes:**")
                if not df_display.empty:
                    if "supporting_documents" in df_display.columns:
                         # Ensure supporting_documents are strings for display
                         df_display.loc[:, "supporting_documents"] = df_display["supporting_documents"].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
                    
                    st.table(df_display)
                else:
                    st.info("Theme data received, but suitable columns for table display were not found or data is empty.")

            else:
                st.info("No themes were extracted or the data format is incorrect.")

        except Exception as e:
            st.error(f"Unexpected error during theme extraction: {e}")
