import uuid

def chunk_text(semantic_blocks, chunk_size=200, overlap=50):
    chunks = []
    current_chunk_text = []
    current_chunk_word_count = 0
    base_meta = None
    
    def finalize_chunk(text_words, meta):
        if not text_words: return
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": " ".join(text_words),
            "page_number": meta["page_number"],
            "section_id": meta["section_id"],
            "section_title": meta["section_title"],
            "chunk_type": meta["chunk_type"],
            "parent_chunk_id": None,
            "footnote_ids": [],
            "cross_ref_ids": []
        })

    for block in semantic_blocks:
        words = block["text"].split()
        
        if base_meta and (
            base_meta["section_id"] != block["section_id"] or 
            base_meta["section_title"] != block["section_title"] or
            base_meta["chunk_type"] != block["chunk_type"]
        ):
            finalize_chunk(current_chunk_text, base_meta)
            current_chunk_text = []
            current_chunk_word_count = 0
            base_meta = None
            
        if not base_meta:
            base_meta = block
            
        while len(words) > 0:
            space_left = chunk_size - current_chunk_word_count
            if len(words) <= space_left:
                current_chunk_text.extend(words)
                current_chunk_word_count += len(words)
                break
            else:
                current_chunk_text.extend(words[:space_left])
                finalize_chunk(current_chunk_text, base_meta)
                
                overlap_words = current_chunk_text[-overlap:] if overlap > 0 else []
                current_chunk_text = overlap_words + words[space_left:]
                current_chunk_word_count = len(current_chunk_text)
                words = []
                
    if current_chunk_text and base_meta:
        finalize_chunk(current_chunk_text, base_meta)
        
    return chunks