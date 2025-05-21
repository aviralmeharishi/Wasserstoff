import pytesseract
from PIL import Image
import pdfplumber
import os

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error reading image: {e}"

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_all_documents(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed_files_count = 0

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        # Ensure consistent .txt extension for output text files
        base_filename, file_extension = os.path.splitext(filename)
        # Use original filename (without .txt) as part of the new .txt filename to keep original name info
        output_txt_filename = base_filename + file_extension.replace('.', '_') + ".txt" 
        output_file = os.path.join(output_dir, output_txt_filename)
        
        text = ""
        if file_extension.lower() in (".png", ".jpg", ".jpeg"):
            print(f"🖼️ Reading Image: {filename}")
            text = extract_text_from_image(file_path)
        elif file_extension.lower() == ".pdf":
            print(f"📄 Reading PDF: {filename}")
            text = extract_text_from_pdf(file_path)
        elif file_extension.lower() == ".txt":
             print(f"📋 Reading TXT: {filename}")
             try:
                with open(file_path, "r", encoding="utf-8") as f_in:
                    text = f_in.read()
             except Exception as e:
                print(f"Error reading TXT file {filename}: {e}")
                text = f"Error reading TXT: {e}"
        else:
            print(f"⚠️ Unsupported file format: {filename} (extension: {file_extension})")
            continue

        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        cleaned_text = "\n".join(cleaned_lines)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"✅ Saved extracted text to: {output_txt_filename}")
        processed_files_count +=1
  
