"""
Medical lexicon and fuzzy matching for OCR post-processing.
Corrects common drug names, dosage units, and medical abbreviations.
"""
from fuzzywuzzy import fuzz, process


# ──────────────────────────────────────────────
# Common Drug Names (expandable)
# ──────────────────────────────────────────────
DRUG_NAMES = [
    "acetaminophen", "acyclovir", "albendazole", "amlodipine", "amoxicillin",
    "aspirin", "atenolol", "atorvastatin", "azithromycin",
    "betamethasone", "bisoprolol", "budesonide",
    "captopril", "carvedilol", "cefalexin", "cefixime", "ceftriaxone",
    "cetirizine", "chlorpheniramine", "ciprofloxacin", "citalopram",
    "clarithromycin", "clindamycin", "clopidogrel", "clotrimazole",
    "codeine", "colchicine",
    "dexamethasone", "diazepam", "diclofenac", "digoxin", "diltiazem",
    "domperidone", "doxycycline",
    "enalapril", "erythromycin", "escitalopram", "esomeprazole",
    "famotidine", "fexofenadine", "fluconazole", "fluoxetine",
    "furosemide",
    "gabapentin", "gentamicin", "glimepiride", "gliclazide",
    "hydrochlorothiazide", "hydroxychloroquine",
    "ibuprofen", "insulin", "ipratropium", "irbesartan",
    "ketoconazole",
    "lactulose", "lansoprazole", "levofloxacin", "lisinopril",
    "loratadine", "losartan",
    "metformin", "methotrexate", "methylprednisolone", "metoclopramide",
    "metoprolol", "metronidazole", "montelukast", "morphine",
    "naproxen", "nifedipine", "nitrofurantoin", "norfloxacin",
    "omeprazole", "ondansetron", "ofloxacin",
    "pantoprazole", "paracetamol", "penicillin", "phenytoin",
    "pioglitazone", "prednisolone", "prednisone", "propranolol",
    "rabeprazole", "ramipril", "ranitidine", "risperidone",
    "salbutamol", "sertraline", "simvastatin", "spironolactone",
    "tamsulosin", "telmisartan", "tramadol", "triamcinolone",
    "valproate", "valsartan", "vancomycin", "verapamil",
    "warfarin",
]

# ──────────────────────────────────────────────
# Dosage Units
# ──────────────────────────────────────────────
UNITS = ["mg", "ml", "mcg", "g", "iu", "units", "drops", "puffs", "tabs", "cap"]

# ──────────────────────────────────────────────
# Frequency Abbreviations
# ──────────────────────────────────────────────
FREQUENCIES = [
    "OD", "BD", "TDS", "QDS", "QID", "SOS", "PRN",
    "once daily", "twice daily", "three times daily",
    "four times daily", "as needed", "before meals",
    "after meals", "at bedtime", "morning", "evening", "night",
    "1-0-1", "1-1-1", "1-0-0", "0-0-1", "0-1-0",
]


def fuzzy_correct(text: str, threshold: int = 80) -> tuple:
    """
    Try to fuzzy-match text against the drug lexicon.

    Args:
        text: OCR output text to correct.
        threshold: Minimum fuzzywuzzy score to accept a match.

    Returns:
        (corrected_text, score, matched_term) or (original_text, 0, None)
    """
    if not text or len(text) < 3:
        return text, 0, None

    text_lower = text.lower().strip()

    # Exact match
    if text_lower in DRUG_NAMES:
        return text_lower, 100, text_lower

    # Fuzzy match against drug names
    match = process.extractOne(text_lower, DRUG_NAMES, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0], match[1], match[0]

    # Try matching against units
    match = process.extractOne(text_lower, UNITS, scorer=fuzz.ratio)
    if match and match[1] >= 90:
        return match[0], match[1], match[0]

    return text, 0, None


def correct_prescription_text(words: list, threshold: int = 80) -> list:
    """
    Correct a list of OCR-output words using fuzzy matching.

    Returns list of dicts with original, corrected, score, and matched_term.
    """
    results = []
    for word in words:
        corrected, score, matched = fuzzy_correct(word, threshold)
        results.append({
            "original": word,
            "corrected": corrected,
            "score": score,
            "matched_term": matched,
            "was_corrected": score >= threshold,
        })
    return results


if __name__ == "__main__":
    # Quick test
    test_words = ["paracetamoll", "amoxicilin", "500", "mg", "twce", "daly"]
    results = correct_prescription_text(test_words)
    for r in results:
        if r["was_corrected"]:
            print(f"  '{r['original']}' → '{r['corrected']}' (score: {r['score']})")
        else:
            print(f"  '{r['original']}' (no match)")
