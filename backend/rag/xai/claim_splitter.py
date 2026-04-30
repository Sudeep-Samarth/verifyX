"""
Splits LLM answer into atomic, independently verifiable claims.
Hard cap: 5 claims max. Rule-based split + optional LLM decomposition (generation only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

ENTITY_PATTERNS = [
    "PSBs",
    "PVBs",
    "FBs",
    "SCBs",
    "All SCBs",
    "NBFCs",
    "UCBs",
]
METRIC_PATTERNS = [
    "CRAR",
    "CET1",
    "GNPA",
    "NNPA",
    "NIM",
    "ROA",
    "ROE",
    "PCR",
    "SLR",
    "LCR",
    "NSFR",
    "leverage ratio",
]
NUMBER_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:per\s*cent|%|bps|lakh\s*crore|crore)", re.IGNORECASE
)


@dataclass
class Claim:
    id: int = 0
    text: str = ""
    is_numerical: bool = False
    entity: Optional[str] = None
    metric: Optional[str] = None
    original_sentence: str = ""


class AtomicClaimSplitter:
    MAX_CLAIMS = 5

    def split(self, answer: str, llm_fn: Optional[Callable[[str], str]] = None) -> List[Claim]:
        raw_sentences = self._sentence_split(answer)
        atomic_with_orig: List[tuple[str, str]] = []
        for sent in raw_sentences:
            if llm_fn and self._is_compound(sent):
                decomposed = self._decompose(sent, llm_fn)
                for d in decomposed:
                    atomic_with_orig.append((d, sent))
            else:
                atomic_with_orig.append((sent, sent))

        claims = [
            self._make_claim(i, t, orig) for i, (t, orig) in enumerate(atomic_with_orig)
        ]
        numerical = [c for c in claims if c.is_numerical]
        non_numerical = [c for c in claims if not c.is_numerical]
        ordered = numerical + non_numerical
        out = ordered[: self.MAX_CLAIMS]
        for i, c in enumerate(out):
            c.id = i
        return out

    def _sentence_split(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [s.strip() for s in sentences if len(s.split()) > 4]

    def _is_compound(self, sent: str) -> bool:
        compound_indicators = [
            r"\band\b.*\band\b",
            r"\bwhile\b",
            r"\bwhereas\b",
            r"\bbut\b",
            r";\s",
        ]
        return any(re.search(p, sent, re.IGNORECASE) for p in compound_indicators)

    def _decompose(self, sent: str, llm_fn: Callable[[str], str]) -> List[str]:
        prompt = (
            "Split this sentence into atomic claims (one fact each). "
            "Return only the claims as a numbered list, nothing else.\n\n"
            f"Sentence: {sent}"
        )
        try:
            result = llm_fn(prompt)
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            cleaned = [re.sub(r"^\d+[\.\)]\s*", "", l) for l in lines]
            return [c for c in cleaned if len(c.split()) > 3][:3]
        except Exception:
            return [sent]

    def _make_claim(self, idx: int, text: str, original: str) -> Claim:
        is_num = bool(NUMBER_PATTERN.search(text))
        entity = next((e for e in ENTITY_PATTERNS if e in text), None)
        metric = next((m for m in METRIC_PATTERNS if m in text), None)
        return Claim(
            id=idx,
            text=text,
            is_numerical=is_num,
            entity=entity,
            metric=metric,
            original_sentence=original,
        )
