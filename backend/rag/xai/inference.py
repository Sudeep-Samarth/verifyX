import json
from typing import Any, Dict, List, Optional, Callable

INFERENCE_PROMPT = """You are a regulatory reasoning engine. 
Your task is to analyze document chunks and infer implicit compliance requirements when explicit mandates are missing.

RULES:
1. DETECT ABSENCE: Summarize what is explicitly NOT found regarding the user query.
2. SIGNAL ANALYSIS: Look for trends, risk concerns (e.g., "fraud increasing"), preventive measures, or monitoring initiatives in the chunks.
3. LOGICAL INFERENCE: If the regulator highlights a specific risk and mentions mitigation, infer the implied necessary control.
4. STRICT LABELING: Never present an inference as a factual mandate. Label it as "Inferred" or "Implied".

OUTPUT FORMAT (JSON):
{{
  "explicit_finding": "Short description of what was NOT found explicitly",
  "inferred_requirement": "Description of the implied compliance control",
  "reasoning_steps": ["Step 1...", "Step 2..."]
}}

CONTEXT CHUNKS:
{context_block}

USER QUERY: {query}

INFERENCE OUTPUT (JSON ONLY):"""

class InferenceEngine:
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def run(self, query: str, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run deductive reasoning on the provided chunks."""
        context_parts = []
        for i, c in enumerate(chunks[:10], start=1):
            txt = (c.get("text") or "").strip()
            context_parts.append(f"[{i}] {txt}")
        
        prompt = INFERENCE_PROMPT.format(
            context_block="\n\n".join(context_parts),
            query=query
        )
        
        response_text = self.llm_fn(prompt)
        if not response_text:
            return None
            
        try:
            # Basic JSON extractor
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
            return None
        except Exception as e:
            print(f"[InferenceEngine] Error parsing response: {e}", flush=True)
            return None
