import streamlit as st
import os
import shutil
import pandas as pd


from backend.mere_functions import process_all_documents
from backend.llm_services import initialize_vector_store, ask_llm, extract_themes_llm

# --- Page Config & Title ---
st.set_page_config(page_title="AI Document Chatbot", layout="wide")
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
        with st.spinner("Knowledge base initialize ho raha hai... (Models load ho rahe hain, thoda time lagega)"):
            try:
                if not os.path.exists(EXTRACTED_TEXT_DIR) or not os.listdir(EXTRACTED_TEXT_DIR):
                    st.warning(f"'{EXTRACTED_TEXT_DIR}' mein process karne ke liye koi text files nahi hain. Pehle files upload aur process karein.")
                    st.session_state.vector_store_initialized = False # Ensure state reflects reality
                else:
                    initialize_vector_store(force_recreate=True)
                    st.session_state.vector_store_initialized = True
                    st.success("Knowledge Base Initialized!")
            except Exception as e:
                st.error(f"Knowledge Base initialize karne mein error: {e}")
                st.session_state.vector_store_initialized = False
    
    if st.session_state.vector_store_initialized:
        st.success("Knowledge Base taiyar hai!")
    else:
        st.info("Knowledge Base abhi initialize nahi hua hai. Kripya files process karke initialize karein.")

    st.markdown("---")
    st.header("📖 Instructions")
    st.markdown("""
    1.  **Document Upload Karo:** Files select karein aur "Save & Process Uploaded Files" button par click karein.
    2.  **Initialize Knowledge Base:** Files process hone ke baad, yahan sidebar mein "Initialize/Re-Initialize Knowledge Base" button par click karein. Ismein thoda time lagega.
    3.  **Sawaal Poocho / Themes Nikalo:** Knowledge base initialize hone ke baad, aap main page par questions pooch sakte hain ya themes extract kar sakte hain.
    """)

# --- Main Page Content ---
# --- Step 1: File Upload & Processing ---
st.header("1. Document Upload & Process Karo")
uploaded_files = st.file_uploader("PDF, PNG, JPG, JPEG, TXT files select karo", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    # Clear previous uploads from DATA_DIR to process only current batch
    for f_name_to_delete in os.listdir(DATA_DIR):
        try:
            os.remove(os.path.join(DATA_DIR, f_name_to_delete))
        except Exception as e_del:
            print(f"Could not delete old file {f_name_to_delete} from {DATA_DIR}: {e_del}")
    # Clear previously extracted texts as well, as we are processing a new batch
    for f_name_to_delete in os.listdir(EXTRACTED_TEXT_DIR):
        try:
            os.remove(os.path.join(EXTRACTED_TEXT_DIR, f_name_to_delete))
        except Exception as e_del:
            print(f"Could not delete old extracted text {f_name_to_delete} from {EXTRACTED_TEXT_DIR}: {e_del}")

    st.info(f"{len(uploaded_files)} file(s) select ki gayi hain. Ab inhe save aur process kiya jayega.")
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"'{uploaded_file.name}' upload karne mein error: {e}")
            # If one file fails to save, maybe stop processing this batch? Or continue?
            # For now, it continues, but the process button will see fewer files in DATA_DIR.

    # Moved processing button to be active only after files are selected
    if st.button("Save Ki Hui Files Ko Process Karo", key="process_button"):
        if not os.listdir(DATA_DIR):
            st.warning("Process karne ke liye file(s) save nahi hui hain. Kripya files select karein.")
        else:
            with st.spinner(f"{len(os.listdir(DATA_DIR))} files process ho rahe hain... (Text extraction)"):
                try:
                    process_all_documents(DATA_DIR, EXTRACTED_TEXT_DIR)
                    st.success(f"Sabhi {len(os.listdir(DATA_DIR))} files process ho gayi! Extracted text '{EXTRACTED_TEXT_DIR}' mein save hua.")
                    st.info("Ab aap Knowledge Base ko initialize/re-initialize kar sakte hain (Sidebar mein button hai).")
                    st.session_state.vector_store_initialized = False # Mark as uninitialized since new texts are processed
                except Exception as e:
                    st.error(f"Files process karne mein error aaya: {e}")

st.markdown("---")

# --- Step 2: Question Answering ---
st.header("2. Sawaal Poocho")
if not st.session_state.vector_store_initialized:
    st.warning("Kripya pehle sidebar se 'Initialize/Re-Initialize Knowledge Base' button par click karein (Files process karne ke baad).")

question = st.text_input("Apna sawaal yahan type karo...", key="question_input_main", disabled=not st.session_state.vector_store_initialized)
if st.button("Sawaal Bhejo", key="ask_button_main", disabled=not st.session_state.vector_store_initialized):
    if not question.strip():
        st.warning("Please pehle sawaal type karo.")
    else:
        with st.spinner("Jawaab dhoondh rahe hain..."):
            try:
                response = ask_llm(question)
                st.markdown("**Jawaab:**")
                st.text_area("Answer", value=response.get("answer", "Koi jawaab nahi mila."), height=150, key="qna_answer_area", disabled=True)
                if response.get("sources"):
                    st.markdown("**Sources (jin files se jawaab mila):**")
                    st.write(", ".join(response.get("sources")))
            except Exception as e:
                st.error(f"Sawaal process karne mein error: {e}")

st.markdown("---")

# --- Step 3: Theme Extraction ---
st.header("3. Themes Extract Karo")
if st.button("Common Themes Nikalo", key="themes_button_main", disabled=not st.session_state.vector_store_initialized):
    with st.spinner("Themes extract kar rahe hain... (AI model thoda samay le sakta hai)"):
        try:
            response_data = extract_themes_llm()
            
            if "error" in response_data:
                st.error(f"Theme extraction mein error aaya: {response_data['error']}")
                if "raw_output" in response_data:
                    st.info("AI ka raw output (agar available ho):")
                    st.text(response_data["raw_output"])
            elif "themes_data" in response_data and response_data["themes_data"]:
                themes_list = response_data["themes_data"]
                df_themes = pd.DataFrame(themes_list)
                
                cols_to_display = ["theme_title", "theme_description", "supporting_documents"]
                # Filter dataframe to only include existing columns from cols_to_display
                df_display = df_themes[[col for col in cols_to_display if col in df_themes.columns]].copy() # Use .copy() to avoid SettingWithCopyWarning
                
                st.markdown("**Identified Themes:**")
                if not df_display.empty:
                    if "supporting_documents" in df_display.columns:
                         df_display.loc[:, "supporting_documents"] = df_display["supporting_documents"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                    
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.info("Themes ka data mila, par table format mein display karne ke liye sahi columns nahi mile ya data khaali hai.")
                    # st.json(themes_list) # Optionally display raw JSON if DataFrame is empty but data exists

            else:
                st.info("Koi themes extract nahi hue ya data format sahi nahi hai.")

        except Exception as e:
            st.error(f"Themes extract karte waqt anumaanit error: {e}")
