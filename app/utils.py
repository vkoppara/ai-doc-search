from docx import Document
from typing import List
import re

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
