import re
from typing import Any, Dict, List

class PassageFilter:
    def __init__(self, min_keyword_overlap: int = 1, min_ce_score: float = 0.1):
        self.min_keyword_overlap = min_keyword_overlap
        self.min_ce_score = min_ce_score
        
        # Domain concept seeds for heuristic classification
        self.domain_seeds = {
            "Fraud/Risk": ["fraud", "prevention", "risk", "hazard", "threat", "mitigation", "scam", "suspicious"],
            "Capital/Banking": ["capital", "cet1", "crar", "slr", "crr", "adequacy", "tier", "liquidity", "lcr"],
            "Payments/UPI": ["upi", "payment", "p2p", "p2m", "wallet", "gateway", "settlement", "transaction"],
            "Investor/Securities": ["investor", "sebi", "securities", "market", "protection", "standard", "asba"],
        }

    def _get_chunk_domain(self, text: str) -> str:
        text_lower = text.lower()
        scores = {d: 0 for d in self.domain_seeds}
        for d, seeds in self.domain_seeds.items():
            for seed in seeds:
                if seed in text_lower:
                    scores[d] += 1
        
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "Other"

    def filter_chunks(self, q_intent: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters chunks based on domain match and keyword overlap.
        q_intent expected keys: 'domain', 'keywords'
        """
        query_domain = q_intent.get("domain", "Other")
        core_keywords = [k.lower() for k in q_intent.get("keywords", [])]
        
        filtered = []
        for c in chunks:
            text = (c.get("text") or "").lower()
            
            # 1. Keyword Overlap
            overlap = sum(1 for kw in core_keywords if kw in text)
            
            # 2. Domain Check
            # If query is in a specific domain, penalize chunks that are clearly in another strict domain
            chunk_domain = self._get_chunk_domain(text)
            
            is_mismatch = False
            # 'Other' and 'General Regulatory' are considered neutral and do not trigger mismatches
            neutral_domains = ["Other", "General Regulatory"]
            if query_domain not in neutral_domains and chunk_domain not in neutral_domains and query_domain != chunk_domain:
                is_mismatch = True
            
            # 3. Decision Logic
            ce_score = c.get("cross_encoder_score", 0.0)
            
            # Rule: Keep if (high overlap) OR (no domain mismatch AND decent CE score)
            # Drift safety: Reject if mismatch is strong or overall relevance is poor
            if is_mismatch and overlap < 2:
                # Strong rejection for domain drift
                continue
            
            if overlap == 0 and ce_score < self.min_ce_score:
                # Rejection for lack of grounding
                continue
            
            c["_filter_info"] = {
                "overlap": overlap,
                "chunk_domain": chunk_domain,
                "is_mismatch": is_mismatch
            }
            filtered.append(c)
            
        return filtered
