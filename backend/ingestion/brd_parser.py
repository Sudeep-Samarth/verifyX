import os
import re
import json
import logging
import pdfplumber
from nltk.tokenize import sent_tokenize

# Suppress pdfminer font parsing warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "test_data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_objectives")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Semantic role keyword map
# Each role maps to trigger words that may appear in a section heading OR
# in the body text of a paragraph. Heading match takes priority.
# ---------------------------------------------------------------------------
ROLE_KEYWORDS: dict[str, list[str]] = {
    "Objective":        ["objective", "goal", "aim", "purpose", "mission", "vision"],
    "Scope":            ["scope", "in scope", "out of scope", "boundary", "inclusions", "exclusions"],
    "Requirement":      ["requirement", "must", "shall", "functional", "non-functional",
                         "feature", "capability", "specification", "user story", "use case"],
    "Constraint":       ["constraint", "limitation", "restriction", "compliance"],
    "Risk":             ["risk", "issue", "mitigation", "threat", "impact", "vulnerability"],
    "Assumption":       ["assumption", "assumed", "presume", "premise"],
    "Dependency":       ["dependency", "depends on", "integration", "interface", "third-party"],
    "Stakeholder":      ["stakeholder", "sponsor", "client", "owner", "user", "actor", "persona"],
    "Background":       ["background", "context", "overview", "introduction", "executive summary",
                         "problem statement", "business problem", "current situation"],
    "Glossary":         ["glossary", "definition", "acronym", "term", "terminology"],
    "Timeline":         ["timeline", "milestone", "schedule", "deadline", "phase", "roadmap",
                         "deliverable", "release"],
    "Budget":           ["budget", "cost", "estimate", "investment", "financial", "expenditure"],
    "Success Criteria": ["success criteria", "acceptance criteria", "kpi", "metric",
                         "key performance", "measurement", "done criteria"],
}

DEFAULT_ROLE = "General"

# Heading detection: numbered sections or ALL-CAPS short lines
HEADING_RE = re.compile(
    r"^(?:\d+\.[\d.]*\s+.+|[A-Z][A-Z\s\-]{4,60}|[IVXLC]+\.\s+.+|Appendix\s+\w+.*)$"
)

# Word styles that python-docx uses for headings
DOCX_HEADING_STYLES = {
    "heading 1", "heading 2", "heading 3", "heading 4",
    "title", "subtitle",
}

# ---------------------------------------------------------------------------
# Helpers shared by all extractors
# ---------------------------------------------------------------------------

def _is_heading(line: str) -> bool:
    """Heuristically decide if a plain text line is a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if len(stripped.split()) > 12:
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+\S", stripped):      # e.g. "1. Scope" or "2.1 Scope"
        return True
    if stripped.isupper() and len(stripped.split()) >= 2:  # ALL-CAPS
        return True
    return False


def _assign_role(heading: str, body: str) -> str:
    """Assign a semantic role. Heading keywords take priority over body."""
    heading_lower = heading.lower()
    body_lower    = body.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in heading_lower for kw in keywords):
            return role
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in body_lower for kw in keywords):
            return role
    return DEFAULT_ROLE


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) < 6:
        return True
    if stripped.isdigit():
        return True
    return False


# ---------------------------------------------------------------------------
# Format-specific raw block extractors
# Each returns list[dict]: {page_number, text, is_heading}
# ---------------------------------------------------------------------------

def _blocks_from_pdf(file_path: str) -> list[dict]:
    """Extract raw blocks from a PDF using pdfplumber."""
    blocks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                raw_text = page.extract_text()
            except Exception:
                continue
            if not raw_text:
                continue
            for line in raw_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                blocks.append({
                    "page_number": page_num,
                    "text": line,
                    "is_heading": _is_heading(line),
                })
    return blocks


def _blocks_from_txt(file_path: str) -> list[dict]:
    """
    Extract raw blocks from a plain-text BRD.
    All content is treated as page 1 (text files have no page concept).
    Blank lines act as paragraph separators.
    """
    blocks = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r").strip()
            if not line:
                continue
            blocks.append({
                "page_number": 1,
                "text": line,
                "is_heading": _is_heading(line),
            })
    return blocks


def _blocks_from_docx(file_path: str) -> list[dict]:
    """
    Extract raw blocks from a .docx file using python-docx.
    Paragraph style names (Heading 1/2/3…) are used first; the heuristic
    _is_heading() is the fallback for un-styled documents.
    """
    try:
        from docx import Document  # lazy import so PDF/TXT paths avoid the dep
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to parse .docx files. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(file_path)
    blocks = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name if para.style else "").lower()
        is_heading  = (style_name in DOCX_HEADING_STYLES) or _is_heading(text)
        blocks.append({
            "page_number": 1,   # python-docx doesn't expose page numbers
            "text": text,
            "is_heading": is_heading,
        })
    return blocks


# Router
_EXTRACTOR = {
    ".pdf":  _blocks_from_pdf,
    ".txt":  _blocks_from_txt,
    ".docx": _blocks_from_docx,
}


def extract_text_blocks(file_path: str) -> list[dict]:
    """Dispatch to the correct extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _EXTRACTOR:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return _EXTRACTOR[ext](file_path)


# ---------------------------------------------------------------------------
# Core semantic record builder (shared by all formats)
# ---------------------------------------------------------------------------

def build_semantic_records(blocks: list[dict]) -> list[dict]:
    """
    Walk raw blocks top-to-bottom:
    - Track current section heading.
    - Flush accumulated paragraph lines when a new heading is encountered.
    - Assign a semantic role to each flushed paragraph.
    - Sentence-tokenise for finer granularity.

    Returns list[dict]: {section_title, semantic_role, text, page_number}
    """
    records = []
    current_heading = "Preamble"
    current_page    = 1
    paragraph_lines: list[str] = []

    def flush(heading: str, page: int, lines: list[str]):
        if not lines:
            return
        body = " ".join(lines)
        try:
            sentences = sent_tokenize(body)
        except Exception:
            sentences = [body]
        role = _assign_role(heading, body)
        for sent in sentences:
            sent = sent.strip()
            if _is_noise(sent):
                continue
            records.append({
                "section_title": heading,
                "semantic_role": role,
                "text":          sent,
                "page_number":   page,
            })

    for block in blocks:
        if block["is_heading"]:
            flush(current_heading, current_page, paragraph_lines)
            paragraph_lines = []
            current_heading = block["text"].strip()
            current_page    = block["page_number"]
        else:
            paragraph_lines.append(block["text"])
            current_page = block["page_number"]

    flush(current_heading, current_page, paragraph_lines)
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_brd(file_path: str) -> list[dict]:
    """
    Full semantic extraction from a BRD file (PDF / TXT / DOCX).

    Returns a list of records:
      {
        "section_title":  str,   # Section heading the text lives under
        "semantic_role":  str,   # Objective / Scope / Requirement / Risk / …
        "text":           str,   # Sentence-level extracted text
        "page_number":    int,   # Always 1 for TXT and DOCX (no page concept)
      }
    """
    blocks  = extract_text_blocks(file_path)
    records = build_semantic_records(blocks)
    return records


def process_single_file(file_path: str, filename: str) -> list[dict]:
    """Extract, label, and write JSON output for a single BRD file."""
    records = parse_brd(file_path)

    base_name   = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_brd_parsed.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    role_counts: dict[str, int] = {}
    for r in records:
        role_counts[r["semantic_role"]] = role_counts.get(r["semantic_role"], 0) + 1

    print(f"✅ Saved: {output_path}")
    print(f"   Total records : {len(records)}")
    print(f"   Roles found   : {role_counts}")
    return records


# Backward-compatible alias (old callers used process_single_pdf)
def process_single_pdf(pdf_path: str, filename: str) -> list[dict]:
    return process_single_file(pdf_path, filename)


def process_all_files():
    """Process every supported BRD file in INPUT_FOLDER."""
    supported = [
        f for f in os.listdir(INPUT_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not supported:
        print(f"No supported BRD files found in {INPUT_FOLDER}")
        return

    for filename in supported:
        file_path = os.path.join(INPUT_FOLDER, filename)
        ext = os.path.splitext(filename)[1].lower()
        print(f"\nProcessing [{ext.upper().lstrip('.')}]: {filename}")
        try:
            process_single_file(file_path, filename)
        except Exception as e:
            print(f"  ⚠️  Skipped {filename}: {e}")


# Backward-compatible alias
def process_all_pdfs():
    process_all_files()


if __name__ == "__main__":
    process_all_files()