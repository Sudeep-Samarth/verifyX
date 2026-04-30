import os
import re
import json
from typing import Any, Dict, List
from groq import Groq
from config import GROQ_API_KEY

EXPANSION_PROMPT = """You are a regulatory and compliance expert.
Your goal is to expand a user's raw query into 2-3 diverse semantic variations to improve search recall in a RAG system containing RBI, SEBI, and bank compliance documents.

Follow these rules:
1. Use regulatory terminology (e.g., "PSB", "SMA", "NPA", "KYC", "AML").
2. Include synonyms and broader/narrower phrasing.
3. Keep each query concise (under 15 words).
4. Return ONLY a bulleted list of the expanded queries.
5. Do NOT include any introductory or concluding text.

INPUT QUERY: {query}

EXPANDED QUERIES:"""

class QueryExpander:
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key
        # Check for key, else use default client behavior
        self.client = Groq(api_key=api_key) if api_key else Groq()

    def expand(self, query: str, max_variants: int = 3) -> List[str]:
        """Generate variations of the query using Groq."""
        if not query or len(query.strip()) < 3:
            return []

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": EXPANSION_PROMPT.format(query=query)}
                ],
                temperature=0.4,
                max_completion_tokens=256,
            )
            content = response.choices[0].message.content or ""
            
            # Parse bulleted list
            lines = content.strip().split("\n")
            variants = []
            for line in lines:
                clean = re.sub(r"^[-*•\d\.]+\s*", "", line).strip()
                if clean and len(clean.split()) > 1:
                    variants.append(clean)
            
            return variants[:max_variants]
        except Exception as e:
            print(f"[QueryExpander] Error: {e}", flush=True)
            return []

    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract core keywords and regulatory domain using Groq."""
        if not query or len(query.strip()) < 3:
            return {"domain": "Other", "keywords": []}

        prompt = f"""Analyze the regulatory query and return a JSON object with:
"domain": One of [Fraud/Risk, Capital/Banking, Payments/UPI, Investor/Securities, Other]
"keywords": List of the most critical 3-5 keywords for strict overlap filtering.

QUERY: {query}

JSON ONLY:"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=100,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            # Validation: Ensure domain is one we recognize
            valid_domains = ["Fraud/Risk", "Capital/Banking", "Payments/UPI", "Investor/Securities", "Other"]
            if data.get("domain") not in valid_domains:
                data["domain"] = "General Regulatory"
            return data
        except Exception as e:
            print(f"[Intent] Fallback keyword-based classification used (Error: {e})", flush=True)
            kws = self.extract_core_terms(query)
            q_lower = query.lower()
            
            # Heuristic mapping
            domain = "General Regulatory"
            if any(w in q_lower for w in ["fraud", "risk", "scam", "monitoring", "hazard"]):
                domain = "Fraud/Risk"
            elif any(w in q_lower for w in ["capital", "cet1", "crar", "adequacy"]):
                domain = "Capital/Banking"
            elif any(w in q_lower for w in ["upi", "payment", "p2p", "transaction"]):
                domain = "Payments/UPI"
            elif any(w in q_lower for w in ["investor", "sebi", "securities"]):
                domain = "Investor/Securities"
            
            return {"domain": domain, "keywords": kws}

    def extract_core_terms(self, query: str) -> List[str]:
        """Heuristic for semantic coverage check."""
        # Simple stopword-filtered keywords
        stopwords = {"what", "are", "the", "for", "in", "and", "is", "of", "how", "much", "to", "with", "a", "an"}
        words = re.findall(r"\b\w{3,}\b", query.lower())
        return [w for w in words if w not in stopwords]
