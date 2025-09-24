from docx import Document
from typing import List, Dict, Any
import re
from pypdf import PdfReader
import pdfplumber
import io
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    texts = [para.text for para in doc.paragraphs if para.text.strip()]

    for table in doc.tables:
        headers = [cell.text for cell in table.rows[0].cells]
        for row in table.rows[1:]:
            first_col = row.cells[0].text.strip() if len(row.cells) > 0 else ''
            for col_idx, cell in enumerate(row.cells):
                if col_idx == 0:
                    continue
                header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx+1}"
                value = cell.text.strip()
                if value:
                    cell_text = f"Table: {header} | Row: {first_col}: | Value: {value}"
                    texts.append(cell_text)
    return '\n'.join(texts)

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using pypdf.
    Tries to preserve simple paragraph seperation. Ignores images.
    """
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            txt = re.sub(r"\n{2,}", "\n\n", txt.strip())
            pages_text.append(txt)
    return "\n\n".join(pages_text)

def extract_rich_pdf_segments(file_path: str, ocr_threshold: int = 40) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]]= []
    ocr_candidates = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages):                
                page_num = page_index + 1
                try:            
                    text = page.extract_text() or ""                    
                except Exception as ex:                    
                    text = ""
                raw_text_len = len(text.strip())                
                if raw_text_len >= ocr_threshold:                    
                    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]                    
                    for para in paras:
                        segments.append({'type': 'text', 'page': page_num, 'content': para})                    
                    ocr_candidates.append(page_index)
                else:        
                    print(f"added to ocr_candidates - {page_index}")            
                    ocr_candidates.append(page_index)                    
                try:                    
                    tables = page.extract_tables() or []
                except Exception as ex:                    
                    tables = []
                has_table_content = False
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = table[0]
                    for row in table[1:]:
                        row_extended = row + ["" for _ in range(len(header)-len(row))]
                        cells = []
                        for h, v in zip(header, row_extended):
                            h_clean = (h or '').strip()
                            v_clean = (v or '').strip()
                            if v_clean:
                                cells.append(f"{h_clean}:{v_clean}" if h_clean else v_clean)
                        if cells:
                            has_table_content = True
                            segments.append({'type': 'table_row', 'page': page_num, 'content': ' | '.join(cells)})                                
                if has_table_content:
                    try:
                        ocr_candidates.remove(page_index)
                    except ValueError:
                        print("Error occurred")
                        pass
                    
        if ocr_candidates:
            for idx in ocr_candidates:
                try:                    
                    img = convert_from_path(file_path, first_page=idx+1, last_page=idx+1)[0]                    
                    ocr_text = pytesseract.image_to_string(img).strip()
                    print(f"ocr-text-{ocr_text}")
                    if ocr_text:
                        segments.append({'type': 'ocr', 'page': idx+1, 'content': ocr_text})
                except Exception as ex:
                    import traceback
                    traceback.print_exc()

                    print(f"exception occurred 2 - {ex}")
                    continue
    except Exception as ex:
        print(f"exception occurred 3- {ex}")
        fallback = extract_text_from_pdf(file_path)
        if fallback.strip():
            segments.append({'type': 'text', 'page': 1, 'content': fallback.strip()})
    return segments

def chunk_segments(segments: List[Dict[str, Any]], max_words: int = 300) -> List[str]:
    chunks: List[str] = []
    current_words = 0
    current_parts: List[str] = []
    current_mode: str | None = None

    def flush():
        nonlocal current_parts, current_words, current_mode
        if current_parts:
            chunks.append('\n'.join(current_parts).strip())
        current_parts = []
        current_words = 0
        current_mode = None

    for seg in segments:
        seg_words = len(seg['content'].split())
        mode = seg['type']
        prefix = f"[PAGE {seg['page']} | {mode.upper()}]"
        line = prefix + seg['content']
        if current_mode is not None and mode !=current_mode:
            flush()
        if current_words + seg_words > max_words and current_parts:
            flush()
        current_parts.append(line)
        current_words += seg_words
        current_mode = mode
    flush()
    return chunks

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len((current+sent).split()) > max_chunk_size:
            chunks.append(current.strip())
            current = sent + " "
        else:
            current += sent + " "
    if current.split():
        chunks.append(current.strip())
    return chunks
