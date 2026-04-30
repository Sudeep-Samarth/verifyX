import fitz
import re

def read_pdf(file_path):
    doc = fitz.open(file_path)
    semantic_blocks = []
    
    current_section_id = ""
    current_section_title = ""
    current_chunk_type = "prose"
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        
        for b in blocks:
            if "lines" not in b:
                continue
                
            block_text = ""
            max_size = 0.0
            is_bold = False
            
            for l in b["lines"]:
                for s in l["spans"]:
                    block_text += s["text"] + " "
                    span_size = s["size"]
                    span_font = s["font"].lower()
                    
                    if span_size > max_size:
                        max_size = span_size
                        
                    if "bold" in span_font or "heavy" in span_font or "black" in span_font:
                        is_bold = True
                        
            block_text = " ".join(block_text.split())
            if not block_text:
                continue
                
            if max_size <= 8.5:
                current_chunk_type = "footnote"
            elif max_size >= 14 or (is_bold and max_size >= 10.5):
                current_chunk_type = "prose"
                if len(block_text) < 150:
                    match = re.match(r"^(\d+\.\d+|\b[A-IVI]+\b|Chapter \w+)\s+(.*)", block_text)
                    if match:
                        current_section_id = match.group(1).strip()
                        current_section_title = match.group(2).strip()
                    else:
                        current_section_title = block_text
            elif block_text.lower().startswith("chart ") or block_text.lower().startswith("table ") or block_text.lower().startswith("box "):
                first_word = block_text.split()[0].lower()
                if first_word in ["chart", "table", "box"]:
                    current_chunk_type = first_word
            else:
                if current_chunk_type != "footnote":
                    current_chunk_type = "prose"
            
            semantic_blocks.append({
                "page_number": page_num + 1,
                "text": block_text,
                "section_id": current_section_id,
                "section_title": current_section_title,
                "chunk_type": current_chunk_type
            })
            
    return semantic_blocks