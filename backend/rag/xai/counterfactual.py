from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional
from .pipeline import XAIPipeline, XAIResult

SIMULATION_PROMPT = """You are a regulatory simulation engine. 
Given a proposed improvement/feature, generate a realistic, high-confidence paragraph as if it were an official policy update from a regulator (like RBI).

FEATURE: {suggestion}

POLICY TEXT:"""

class CounterfactualEngine:
    def __init__(self, pipeline: XAIPipeline, llm_fn: Optional[Callable[[str], str]] = None):
        self.pipeline = pipeline
        self.llm_fn = llm_fn

    def simulate_impact(
        self, 
        query: str, 
        original_chunks: List[Dict], 
        suggestions: List[str]
    ) -> List[Dict[str, Any]]:
        if not suggestions or not self.llm_fn:
            return []
            
        results = []
        for sugg in suggestions:
            print(f"[Counterfactual] Simulating impact of: {sugg}...", flush=True)
            
            # 1. Generate Pseudo-chunk
            prompt = SIMULATION_PROMPT.format(suggestion=sugg)
            pseudo_text = self.llm_fn(prompt)
            
            pseudo_chunk = {
                "chunk_id": f"sim_{hash(sugg) % 10000}",
                "text": pseudo_text,
                "report_type": "Simulated Compliance Update",
                "edition_date": "2026-Projected",
                "section_id": "SIM-01",
                "section_title": "Projected Capabilities",
                "is_pseudo": True,
                "doc_id": "Simulated Update (2026)",
                "section": "SIM-01 Projected Capabilities"
            }
            
            # 2. Inject into context
            sim_chunks = [pseudo_chunk] + original_chunks
            
            # 3. Re-run Generation (Simple grounding)
            from .assistant import REPROMPT_TEMPLATE
            
            context_parts = []
            for i, c in enumerate(sim_chunks, start=1):
                txt = (c.get("text") or "").strip()
                context_parts.append(f"[{i}] {txt}")
                
            reprompt = REPROMPT_TEMPLATE.format(
                context_block="\n\n".join(context_parts),
                query=query
            )
            
            sim_answer = self.llm_fn(reprompt)
            
            # 4. Run XAI on simulated state
            sim_res = self.pipeline.run(query, sim_answer, sim_chunks)
            
            results.append({
                "suggestion": sugg,
                "impact_verdict": sim_res.verdict.gate.value,
                "impact_score": sim_res.verdict.confidence,
                "simulated_answer": sim_answer,
                "pseudo_passage": pseudo_text
            })
            
        return results
