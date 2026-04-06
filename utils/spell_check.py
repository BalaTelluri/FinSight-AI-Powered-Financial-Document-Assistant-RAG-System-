from __future__ import annotations
import re
from groq import Groq
import os

# Common financial typos — instant fix without API call
QUICK_FIXES = {
    "insurence": "insurance", "insuranc": "insurance", "insruance": "insurance",
    "helth": "health", "healt": "health", "halth": "health",
    "baning": "banking", "bnaking": "banking", "bankng": "banking",
    "motgage": "mortgage", "mortage": "mortgage", "morggage": "mortgage",
    "laon": "loan", "lona": "loan", "lona": "loan",
    "benfit": "benefit", "benifit": "benefit", "benefti": "benefit",
    "exclsion": "exclusion", "excluion": "exclusion",
    "maternty": "maternity", "materntiy": "maternity",
    "premuim": "premium", "primeum": "premium",
    "covrage": "coverage", "coverge": "coverage",
    "reimburse": "reimburse", "reimburs": "reimburse",
    "dentl": "dental", "detal": "dental",
    "hosptial": "hospital", "hopsital": "hospital",
    "vehicl": "vehicle", "vehcile": "vehicle",
    "accidnt": "accident", "acident": "accident",
    "repayemnt": "repayment", "repayemnt": "repayment",
    "eligbility": "eligibility", "eligiblity": "eligibility",
}

def quick_fix(text: str) -> str:
    """Fix common typos instantly without API call."""
    words = text.split()
    fixed = [QUICK_FIXES.get(w.lower(), w) for w in words]
    return " ".join(fixed)

def is_too_short_or_vague(text: str) -> bool:
    """Detect if input is too short or vague to answer."""
    text = text.strip().lower()
    vague = ["ok", "okay", "yes", "no", "maybe", "sure", "hmm", "what", "how", "why", "tell me"]
    if len(text.split()) <= 1 and text in vague:
        return True
    if len(text) < 5:
        return True
    return False

def needs_clarification(text: str) -> bool:
    """Detect if the question is too ambiguous."""
    text = text.strip().lower()
    # Very short with no financial keywords
    financial_keywords = [
        "insurance", "cover", "loan", "bank", "health", "car", "home",
        "premium", "claim", "benefit", "maternity", "dental", "accident",
        "repay", "interest", "eligible", "policy", "exclusion", "limit"
    ]
    if len(text.split()) <= 2:
        has_keyword = any(kw in text for kw in financial_keywords)
        if not has_keyword:
            return True
    return False

def correct_spelling(text: str, client: Groq) -> tuple[str, bool]:
    """
    Returns (corrected_text, was_corrected)
    Uses quick fix first, then LLM if needed.
    """
    # Step 1: Quick fix
    fixed = quick_fix(text)

    # Step 2: Check if it looks like it has typos
    has_typo = _looks_like_typo(text)

    if not has_typo:
        return fixed, False

    # Step 3: Use LLM to correct
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a spell checker for financial questions. Fix ONLY spelling mistakes. Return ONLY the corrected sentence — nothing else. If no corrections needed, return the original sentence exactly."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=100
        )
        corrected = response.choices[0].message.content.strip()
        was_corrected = corrected.lower() != text.lower()
        return corrected, was_corrected
    except:
        return fixed, fixed.lower() != text.lower()

def _looks_like_typo(text: str) -> bool:
    """Simple heuristic to detect possible typos."""
    words = text.split()
    for word in words:
        word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(word_clean) > 3 and word_clean in QUICK_FIXES:
            return True
        # Detect repeated characters like "heealth", "bankkk"
        if re.search(r'(.)\1{2,}', word_clean):
            return True
    return False
