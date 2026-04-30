from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional
from .pipeline import XAIPipeline, XAIResult
from .aggregator import TrustGate

REPROMPT_TEMPLATE = """You are a compliance assistant. The previous answer contained claims that could not be fully verified against the source documents. 

REWRITE the answer using ONLY the following verified context passages. 
- If a claim cannot be directly supported by these passages, DO NOT include it.
- Maintain a strictly formal and compliant tone.
- Cite your sources inline using [n].

CONTEXT:
{context_block}

QUESTION: {query}
"""

EVOLUTIONARY_SYNTHESIS_TEMPLATE = """You are a regulatory compliance assistant. The previous answer was INCOMPLETE because it ignored a more recent regulatory update that was available in the retrieved context.

REWRITE the answer to include ALL relevant time periods found below.
REQUIREMENTS:
- Lead with the MOST RECENT data point (with "As of [date]..." qualifier).
- Then describe the EARLIER context or evolution.
- Note when data is segmented (e.g., PSBs vs PVBs) vs. aggregate (all SCBs).
- Cite every fact inline with [n].
- Do NOT invent any data not present in the context.

CONTEXT:
{context_block}

ORIGINAL QUESTION: {query}

OLDER EDITION IGNORED: {older_edition}
NEWER EDITION MISSING FROM ANSWER: {newer_edition}
"""

class XAIAssistant:
    def __init__(self, pipeline: XAIPipeline, llm_fn: Optional[Callable[[str], str]] = None):
        """
        pipeline: The XAIPipeline instance to run verification.
        llm_fn: A function that takes a prompt and returns an LLM completion string.
        """
        self.pipeline = pipeline
        self.llm_fn = llm_fn

    def run_with_autofix(
        self, 
        query: str, 
        initial_answer: str, 
        chunks: List[Dict],
        max_retries: int = 1
    ) -> XAIResult:
        autofix_enabled = os.getenv("XAI_AUTOFIX", "0").lower() in ("1", "true", "yes")
        
        # Pass 1: Initial Verification
        res = self.pipeline.run(query, initial_answer, chunks)
        
        if not autofix_enabled:
            return res
            
        # Trigger condition: Non-Compliant or Needs Review with low confidence
        needs_fix = (
            res.verdict.gate != TrustGate.SAFE or 
            res.verdict.frac_unattributed > 0.3 or
            res.verdict.hallucination_detected
        )
        
        if needs_fix and self.llm_fn and max_retries > 0:
            print(f"[XAI Assistant] Low trust detected (Gate={res.verdict.gate.value}). Triggering Auto-Fix...", flush=True)
            
            # Prepare re-grounding context
            context_parts = []
            for i, c in enumerate(chunks, start=1):
                txt = (c.get("text") or "").strip()
                context_parts.append(f"[{i}] {txt}")
            
            reprompt = REPROMPT_TEMPLATE.format(
                context_block="\n\n".join(context_parts),
                query=query
            )
            
            # Step 2: Regenerate grounded answer
            new_answer = self.llm_fn(reprompt)
            if not new_answer:
                return res # Fallback to original
                
            print(f"[XAI Assistant] Regenerated answer (len={len(new_answer)}). Re-verifying...", flush=True)
            
            # Step 3: Re-verify
            final_res = self.pipeline.run(query, new_answer, chunks)
            
            # Append meta-info about the fix
            final_res.artifact["autofix_triggered"] = True
            final_res.artifact["original_gate"] = res.verdict.gate.value
            final_res.artifact["original_answer"] = initial_answer
            
            res = final_res

        # --- Temporal Completeness / Evolutionary Synthesis Pass ---
        # Trigger: answer is safe/truthful BUT ignored newer edition in context
        evolutionary_enabled = os.getenv("XAI_EVOLUTIONARY_SYNTHESIS", "1").lower() in ("1", "true", "yes")
        num_ignored = getattr(res.verdict, "num_complementary_ignored", 0)
        if evolutionary_enabled and num_ignored > 0 and self.llm_fn:
            complementary = getattr(res.artifact.get("conflict_report", {}), "complementary_updates", None)
            # Get from the report directly via the artifact
            comp_updates = res.artifact.get("complementary_updates", [])
            # Fall back: get from the pipeline's conflict report stored in artifact
            if not comp_updates:
                comp_updates = []

            # Find the first ignored newer edition
            ignored = [c for c in comp_updates if not c.get("answer_includes_newer", True)]
            if ignored:
                first_ignored = ignored[0]
                older_ed = first_ignored.get("older_edition", "")
                newer_ed = first_ignored.get("newer_edition", "")
                
                print(
                    f"[XAI Assistant] Temporal Incompleteness: answer ignored {newer_ed}. "
                    f"Triggering Evolutionary Synthesis...", flush=True
                )
                
                context_parts = []
                for i, c in enumerate(chunks, start=1):
                    txt = (c.get("text") or "").strip()
                    context_parts.append(f"[{i}] {txt}")
                
                synth_prompt = EVOLUTIONARY_SYNTHESIS_TEMPLATE.format(
                    context_block="\n\n".join(context_parts),
                    query=query,
                    older_edition=older_ed,
                    newer_edition=newer_ed,
                )
                
                synth_answer = self.llm_fn(synth_prompt)
                if synth_answer:
                    print(f"[XAI Assistant] Evolutionary synthesis complete. Re-verifying...", flush=True)
                    final_res = self.pipeline.run(query, synth_answer, chunks)
                    final_res.artifact["evolutionary_synthesis_triggered"] = True
                    final_res.artifact["older_edition_in_context"] = older_ed
                    final_res.artifact["newer_edition_synthesized"] = newer_ed
                    final_res.artifact["original_answer"] = initial_answer
                    res = final_res
        
        # --- Inference Phase ---
        inference_enabled = os.getenv("XAI_INFERENCE", "0").lower() in ("1", "true", "yes")
        if inference_enabled and res.verdict.gate != TrustGate.SAFE and self.llm_fn:
            from .inference import InferenceEngine
            engine = InferenceEngine(self.llm_fn)
            inference_res = engine.run(query, chunks)
            if inference_res:
                res.artifact["inferred_findings"] = inference_res
                
        return res
