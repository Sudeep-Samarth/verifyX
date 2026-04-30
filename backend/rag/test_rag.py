from retriever import get_hybrid_rag_results

def test_query():
    query = "What are the new guidelines for the CET1 ratio?"
    
    print(f"\n--- RAG Testing Execution ---")
    print(f"User Query: '{query}'\n")
    
    try:
        final_chunks = get_hybrid_rag_results(query, mode="query")
        
        print("\n\n=== VERIFIED TOP RESULTS ===")
        for i, chunk in enumerate(final_chunks):
            print(f"\n[Rank {i+1}] Cross-Encoder Score: {chunk.get('cross_encoder_score', 0):.4f} (RRF: {chunk.get('rrf_score', 0):.4f})")
            print(f"Source: {chunk.get('report_type')} {chunk.get('edition_date')} | Section: {chunk.get('section_id')} - {chunk.get('section_title')}")
            # Snippet of text
            text = chunk.get('text', '')
            print(f"Text Snippet: {text[:250]}...")
            
    except Exception as e:
        print(f"RAG Failed: {e}")

if __name__ == "__main__":
    test_query()
