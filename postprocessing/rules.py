"""
Regex-based extraction for dosage, frequency, and duration patterns.
"""
import re


# ──────────────────────────────────────────────
# Dosage patterns: "500 mg", "250mg", "0.5 g"
# ──────────────────────────────────────────────
DOSAGE_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*(mg|ml|mcg|g|iu|units?|drops?|puffs?|tabs?|caps?|syp|inj|susp)',
    re.IGNORECASE
)

# ──────────────────────────────────────────────
# Frequency patterns: "1-0-1", "twice daily", "BD", "TDS"
# ──────────────────────────────────────────────
FREQUENCY_PATTERNS = [
    (re.compile(r'(\d)-(\d)-(\d)', re.IGNORECASE), "schedule"),
    (re.compile(r'(once|twice|thrice|three times|four times)\s*(daily|a day|per day)',
                re.IGNORECASE), "frequency"),
    (re.compile(r'\b(OD|BD|TDS|QDS|QID|PRN|SOS|BID|TID|HS|AC|PC|STAT|OW|OM|ON)\b',
                re.IGNORECASE), "abbreviation"),
    (re.compile(r'(before|after|with)\s*(meals?|food|breakfast|lunch|dinner)',
                re.IGNORECASE), "timing"),
    (re.compile(r'(at|every)\s*(bedtime|night|morning|evening)',
                re.IGNORECASE), "timing"),
    (re.compile(r'\b(if\s+needed|when\s+required|as\s+required|as\s+needed)\b',
                re.IGNORECASE), "prn"),
    (re.compile(r'\b(immediately|right\s+away|right\s+now)\b',
                re.IGNORECASE), "stat"),
]

# ──────────────────────────────────────────────
# Duration patterns: "5 days", "2 weeks", "for 7 days"
# ──────────────────────────────────────────────
DURATION_PATTERN = re.compile(
    r'(?:for\s+)?(\d+)\s*(days?|weeks?|months?)',
    re.IGNORECASE
)


def extract_dosage(text: str) -> list:
    """Extract dosage information from text."""
    matches = DOSAGE_PATTERN.findall(text)
    return [{"value": m[0], "unit": m[1].lower()} for m in matches]


def extract_frequency(text: str) -> list:
    """Extract frequency/timing information from text."""
    results = []
    for pattern, pattern_type in FREQUENCY_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            match_text = m if isinstance(m, str) else " ".join(m)
            results.append({"text": match_text.strip(), "type": pattern_type})
    return results


def extract_duration(text: str) -> list:
    """Extract duration information from text."""
    matches = DURATION_PATTERN.findall(text)
    return [{"value": m[0], "unit": m[1].lower()} for m in matches]


def extract_all(text: str) -> dict:
    """Extract all prescription components from text."""
    return {
        "dosage": extract_dosage(text),
        "frequency": extract_frequency(text),
        "duration": extract_duration(text),
        "raw_text": text,
    }


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Paracetamol 500 mg twice daily for 5 days",
        "Amoxicillin 250mg 1-0-1 after meals 7 days",
        "Insulin 10 units BD before meals",
    ]
    for text in test_texts:
        result = extract_all(text)
        print(f"\n  Input: '{text}'")
        print(f"  Dosage:    {result['dosage']}")
        print(f"  Frequency: {result['frequency']}")
        print(f"  Duration:  {result['duration']}")
