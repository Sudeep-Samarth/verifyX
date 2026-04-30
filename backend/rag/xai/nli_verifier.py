# Maynez et al. (2020) ACL; Laurer et al. (2023) EMNLP (DeBERTa NLI)
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_NLI_FORWARD_LOCK = threading.Lock()

STAGE1_MODEL = os.getenv("XAI_NLI_MODEL", "cross-encoder/nli-deberta-v3-large")
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("XAI_NLI_HIGH_CONF", "0.90"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("XAI_NLI_LOW_CONF", "0.50"))


@dataclass
class ClaimVerdict:
    claim_id: int
    claim_text: str
    source_passage: str
    label: str
    confidence: float
    stage: int
    reasoning: str
    is_hallucination: bool
    second_pass_used: bool


@dataclass
class NLIResult:
    verdicts: List[ClaimVerdict]
    answer_entailment_score: float
    failed_claims: List[str]
    hallucination_detected: bool


class TwoStageNLIVerifier:
    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        second_pass_fn: Optional[Callable[[str], List[str]]] = None,
        model_id: Optional[str] = None,
    ):
        self._tokenizer = None
        self._model = None
        self._device: Optional[torch.device] = None
        self.model_id = (model_id or STAGE1_MODEL).strip()
        self.llm_fn = llm_fn
        self.second_pass_fn = second_pass_fn

    def _pick_device(self) -> torch.device:
        raw = (os.getenv("XAI_NLI_DEVICE") or os.getenv("CE_DEVICE") or "auto").strip().lower()
        if raw == "cpu":
            return torch.device("cpu")
        if raw == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if raw == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._device = self._pick_device()
        local_only = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=local_only
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, local_files_only=local_only
        )
        self._model.to(self._device)
        self._model.eval()
        self._id2label = self._model.config.id2label
        if not isinstance(self._id2label, dict):
            self._id2label = dict(self._id2label)

    def stage1_label_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Public: premise=source text, hypothesis=claim (for edition NLI hooks)."""
        with _NLI_FORWARD_LOCK:
            self._load_model()
            assert self._tokenizer is not None and self._model is not None and self._device is not None
            inputs = self._tokenizer(
                (premise or "")[:2000],
                hypothesis or "",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
            out: Dict[str, float] = {}
            for i in range(len(probs)):
                lab = str(self._id2label.get(i, str(i))).lower()
                out[lab] = probs[i].item()
            return out

    def _stage1_score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        return self.stage1_label_scores(premise, hypothesis)

    def _best_label(self, scores: Dict[str, float]) -> tuple[str, float]:
        if not scores:
            return "neutral", 0.0
        label = max(scores, key=lambda k: scores[k])
        return label, scores[label]

    def _entailment_strength(self, scores: Dict[str, float]) -> float:
        for k, v in scores.items():
            if "entail" in k:
                return float(v)
        return 0.0

    def _stage2_arbitrate(self, claim: str, passage: str) -> tuple[str, float, str]:
        if self.llm_fn is None:
            return "neutral", 0.60, "No LLM available for stage 2"

        prompt = f"""You are a financial regulatory compliance expert analyzing RBI FSR documents.

Given a SOURCE PASSAGE and a CLAIM, determine if the passage supports, contradicts, or is neutral to the claim.
Pay attention to: conditional statements, percentage ranges, time periods, bank cohort specifics (PSBs/PVBs/FBs).

SOURCE PASSAGE:
{passage}

CLAIM:
{claim}

Respond in this exact format:
LABEL: <entailment|neutral|contradiction>
CONFIDENCE: <0.0-1.0>
REASONING: <one sentence explaining why>"""

        try:
            response = self.llm_fn(prompt)
            lines = [l for l in response.strip().split("\n") if ":" in l]
            parsed: Dict[str, str] = {}
            for line in lines:
                key, _, rest = line.partition(":")
                parsed[key.strip().upper()] = rest.strip()
            label = parsed.get("LABEL", "neutral").lower().strip()
            confidence = float(parsed.get("CONFIDENCE", "0.65").split()[0])
            reasoning = parsed.get("REASONING", "LLM arbitration")
            if label not in ("entailment", "neutral", "contradiction"):
                label = "neutral"
            return label, min(max(confidence, 0.0), 1.0), reasoning
        except Exception:
            return "neutral", 0.60, "LLM arbitration parse error"

    def _second_pass(self, claim_text: str, original_verdict: ClaimVerdict) -> ClaimVerdict:
        if self.second_pass_fn is None:
            original_verdict.is_hallucination = original_verdict.label == "contradiction"
            return original_verdict

        try:
            fresh_chunks = self.second_pass_fn(claim_text)
            if not fresh_chunks:
                original_verdict.is_hallucination = True
                return original_verdict

            best_label, best_conf = "neutral", 0.0
            best_passage = original_verdict.source_passage or ""

            for chunk in fresh_chunks[:3]:
                scores = self._stage1_score(chunk, claim_text)
                label, conf = self._best_label(scores)
                nl = label
                if "entail" in nl:
                    nl = "entailment"
                if nl == "entailment" and conf > best_conf:
                    best_label, best_conf = "entailment", conf
                    best_passage = chunk[:300]

            if best_label == "entailment" and best_conf >= LOW_CONFIDENCE_THRESHOLD:
                original_verdict.label = "entailment"
                original_verdict.confidence = best_conf
                original_verdict.source_passage = best_passage
                original_verdict.reasoning = "Verified on second-pass retrieval"
                original_verdict.second_pass_used = True
                original_verdict.is_hallucination = False
            else:
                original_verdict.is_hallucination = original_verdict.label == "contradiction"
                original_verdict.second_pass_used = True

        except Exception:
            original_verdict.is_hallucination = False

        return original_verdict

    def verify_claim(self, claim: Any, passage: str) -> ClaimVerdict:
        if not passage or not str(passage).strip():
            return ClaimVerdict(
                claim_id=claim.id,
                claim_text=claim.text,
                source_passage="",
                label="neutral",
                confidence=0.0,
                stage=0,
                reasoning="No source passage available",
                is_hallucination=False,
                second_pass_used=False,
            )

        scores = self._stage1_score(passage, claim.text)
        label, conf = self._best_label(scores)
        norm_label = label
        if "entail" in norm_label:
            norm_label = "entailment"
        elif "contrad" in norm_label:
            norm_label = "contradiction"
        else:
            norm_label = "neutral"

        if conf >= HIGH_CONFIDENCE_THRESHOLD:
            return ClaimVerdict(
                claim_id=claim.id,
                claim_text=claim.text,
                source_passage=passage[:300],
                label=norm_label,
                confidence=conf,
                stage=1,
                reasoning=f"Stage 1 high-confidence: {conf:.3f}",
                is_hallucination=(
                    norm_label == "contradiction" and conf >= HIGH_CONFIDENCE_THRESHOLD
                ),
                second_pass_used=False,
            )

        if LOW_CONFIDENCE_THRESHOLD <= conf < HIGH_CONFIDENCE_THRESHOLD:
            label2, conf2, reasoning2 = self._stage2_arbitrate(claim.text, passage)
            verdict = ClaimVerdict(
                claim_id=claim.id,
                claim_text=claim.text,
                source_passage=passage[:300],
                label=label2,
                confidence=conf2,
                stage=2,
                reasoning=reasoning2,
                is_hallucination=False,
                second_pass_used=False,
            )
        else:
            verdict = ClaimVerdict(
                claim_id=claim.id,
                claim_text=claim.text,
                source_passage=passage[:300],
                label=norm_label,
                confidence=conf,
                stage=1,
                reasoning=f"Low confidence: {conf:.3f}",
                is_hallucination=False,
                second_pass_used=False,
            )

        if verdict.label in ("neutral", "contradiction"):
            verdict = self._second_pass(claim.text, verdict)

        return verdict

    def verify_all(self, claims: List[Any], attributions: List[Any]) -> NLIResult:
        attr_map = {a.claim_id: a for a in attributions}
        verdicts: List[ClaimVerdict] = []

        for claim in claims:
            attr = attr_map.get(claim.id)
            passage = (
                (attr.source_passage or "")
                if (attr and attr.is_attributed)
                else ""
            )
            verdicts.append(self.verify_claim(claim, passage))

        def claim_score(v: ClaimVerdict) -> float:
            if v.label == "entailment":
                return 1.0 * v.confidence
            if v.label == "neutral":
                return 0.3 * v.confidence
            return -1.0 * v.confidence

        scores = [claim_score(v) for v in verdicts]
        answer_score = sum(scores) / len(scores) if scores else 0.0
        answer_score = max(0.0, min(1.0, (answer_score + 1.0) / 2.0))

        failed = [v.claim_text for v in verdicts if v.label != "entailment"]
        hallucination = any(v.is_hallucination for v in verdicts)

        return NLIResult(
            verdicts=verdicts,
            answer_entailment_score=answer_score,
            failed_claims=failed,
            hallucination_detected=hallucination,
        )


# --- Sentence-level baseline (single-shot NLI vs chunks) for smoke tests / fast checks ---
@dataclass
class SentenceVerdict:
    sentence: str
    max_entailment_score: float
    best_chunk_index: int
    label: str


@dataclass
class SentenceNLIResult:
    verdicts: List[SentenceVerdict]
    answer_entailment_score: float
    failed_sentences: List[str]


class SentenceLevelNLIVerifier:
    """Original sentence × chunk max-entailment (deterministic; used in pytest)."""

    def __init__(self, model_id: Optional[str] = None):
        self._two = TwoStageNLIVerifier(
            llm_fn=None, second_pass_fn=None, model_id=model_id
        )

    def verify(self, answer: str, chunks: List[str]) -> SentenceNLIResult:
        self._two._load_model()
        sentences = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
        sentences = [s for s in sentences if len(s.split()) > 5]
        verdicts: List[SentenceVerdict] = []
        for sent in sentences:
            best_score = 0.0
            best_idx = -1
            for i, chunk in enumerate(chunks):
                scores = self._two.stage1_label_scores(chunk, sent)
                ent = 0.0
                for k, v in scores.items():
                    if "entail" in k:
                        ent = max(ent, v)
                if ent > best_score:
                    best_score = ent
                    best_idx = i
            if best_score >= 0.75:
                lab = "entailed"
            elif best_score >= 0.5:
                lab = "neutral"
            else:
                lab = "contradiction"
            verdicts.append(
                SentenceVerdict(sent, best_score, best_idx, lab)
            )
        sc = [v.max_entailment_score for v in verdicts]
        ans = sum(sc) / len(sc) if sc else 0.0
        failed = [v.sentence for v in verdicts if v.label != "entailed"]
        return SentenceNLIResult(verdicts, ans, failed)


NLIVerifier = TwoStageNLIVerifier
