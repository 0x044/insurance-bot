import fitz  # PyMuPDF
import os
import re
from pathlib import Path

def clean_text(text):
    """Clean extracted text to improve quality"""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR/extraction issues
    text = text.replace('|', 'I')  # Common OCR mistake
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
    
    # Remove header/footer patterns (customize based on your documents)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Insurance Policy Document.*\d{4}', '', text)
    
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with improved layout handling"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            # Get text with preservation of reading order and tables
            text = page.get_text("text")
            
            # Clean the text
            clean = clean_text(text)
            
            # Add page number reference
            full_text.append(f"[Page {page_num + 1}]\n{clean}")
        
        # Join all pages
        return "\n\n".join(full_text)
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def load_knowledge_base(directory="data"):
    """Load all PDFs from a directory"""
    directory_path = Path(directory)
    all_text = []
    
    if not directory_path.exists():
        os.makedirs(directory, exist_ok=True)
        return ""
    
    for pdf_file in directory_path.glob("*.pdf"):
        try:
            text = extract_text_from_pdf(str(pdf_file))
            all_text.append(f"[Document: {pdf_file.name}]\n{text}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    return "\n\n".join(all_text)