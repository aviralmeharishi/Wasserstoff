import streamlit as st
import os
import shutil

from backend.mere_functions import process_all_documents
from backend.llm_services import initialize_vector_store, ask_llm, extract_themes_llm

st.set_page_config(page_title="AI Document Chatbot", layout="wide")
st.title("📄 AI Document Chatbot & Theme Identifier")
st.caption("Internship Task Submission")

DATA_DIR = "data_streamlit_uploads"
EXTRACTED_TEXT_DIR = "extracted_texts_streamlit"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

if st.sidebar.button("Initialize/Re-Initialize Knowledge Base (from processed texts)", key="init_kb"):
    with st.spinner("Knowledge base initialize ho raha hai... (Models load ho rahe hain, thoda time lagega)"):
        try:
            initialize_vector_store(force_recreate=True)
            st.session_state.vector_store_initialized = True
            st.sidebar.success("Knowledge Base Initialized!")
        except Exception as e:
            st.sidebar.error(f"Knowledge Base initialize karne mein error: {e}")
            st.session_state.vector_store_initialized = False

st.header("1. Document Upload Karo aur Process Karo")
uploaded_files = st.file_uploader("PDF, PNG, JPG, JPEG, TXT files select karo", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_files_processed_successfully = True
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"'{uploaded_file.name}' upload ho gaya hai.")
        except Exception as e:
            st.error(f"'{uploaded_file.name}' upload karne mein error: {e}")
            all_files_processed_successfully = False
            continue

    if st.button("Sabhi Uploaded Files Ko Process Karo", key="process_button"):
        if not os.listdir(DATA_DIR):
            st.warning("Process karne ke liye कृपया pehle files upload karein.")
        else:
            with st.spinner(f"{len(os.listdir(DATA_DIR))} files process ho rahe hain... (Text extraction)"):
                try:
                    process_all_documents(DATA_DIR, EXTRACTED_TEXT_DIR)
                    st.success(f"Sabhi {len(os.listdir(DATA_DIR))} files process ho gaye! Extracted text '{EXTRACTED_TEXT_DIR}' mein save hua.")
                    st.info("Ab aap Knowledge Base ko initialize/re-initialize kar sakte hain (Sidebar mein button hai).")
                except Exception as e:
                    st.error(f"Files process karne mein error aaya: {e}")

st.markdown("---")

st.header("2. Sawaal Poocho")
if not st.session_state.vector_store_initialized:
    st.warning("Kripya pehle sidebar se 'Initialize/Re-Initialize Knowledge Base' button par click karein.")

question = st.text_input("Apna sawaal yahan type karo...", key="question_input", disabled=not st.session_state.vector_store_initialized)
if st.button("Sawaal Bhejo", key="ask_button", disabled=not st.session_state.vector_store_initialized):
    if not question.strip():
        st.warning("Please pehle sawaal type karo.")
    else:
        with st.spinner("Jawaab dhoondh rahe hain..."):
            try:
                response = ask_llm(question)
                st.markdown("**Jawaab:**")
                st.write(response.get("answer", "Koi jawaab nahi mila."))
                if response.get("sources"):
                    st.markdown("**Sources (jin files se jawaab mila):**")
                    st.write(", ".join(response.get("sources")))
            except Exception as e:
                st.error(f"Sawaal process karne mein error: {e}")
                st.info("Kya aapne Knowledge Base initialize kiya hai?")

st.markdown("---")

st.header("3. Themes Extract Karo")
if st.button("Common Themes Nikalo", key="themes_button", disabled=not st.session_state.vector_store_initialized):
    with st.spinner("Themes extract kar rahe hain..."):
        try:
            response = extract_themes_llm()
            st.markdown("**Identified Themes:**")
            st.text_area("Themes", value=response.get("themes_text", "Koi themes nahi mile."), height=200, disabled=True)
            if response.get("supporting_documents"):
                st.markdown("**Supporting Documents (jin par themes based hain):**")
                st.write(", ".join(response.get("supporting_documents")))
        except Exception as e:
            st.error(f"Themes extract karne mein error: {e}")
            st.info("Kya aapne Knowledge Base initialize kiya hai?")

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  **Document Upload Karo:** Files select karke "Sabhi Uploaded Files Ko Process Karo" par click karein.
2.  **Initialize Knowledge Base:** File process hone ke baad, yahan sidebar mein "Initialize/Re-Initialize Knowledge Base" button par click karein. Ismein thoda time lagega.
3.  **Sawaal Poocho:** Knowledge base initialize hone ke baad, question type karke "Sawaal Bhejo" par click karein.
4.  **Themes Extract Karo:** Themes nikalne ke liye button par click karein.
""")
