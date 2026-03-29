"""
Build a bigram ARPA-format language model from the medical lexicon.

This produces a file that pyctcdecode + kenlm can consume for beam-search
decoding.  Drug names, dosage units, and frequency terms are given uniform
log-probabilities, with drug→unit and drug→frequency bigrams.

Usage:
    python postprocessing/build_lm.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from postprocessing.lexicon import DRUG_NAMES, UNITS, FREQUENCIES


def build_arpa(output_path: str):
    import pandas as pd
    import subprocess
    import shutil
    
    # ── 1. Create corpus from lexicon and manifest ──
    corpus_path = "/tmp/lm_corpus.txt"
    lines = []
    
    # Add from lexicon to ensure vocabulary is preserved
    for name in DRUG_NAMES:
        lines.append(name.lower())
    for u in UNITS:
        lines.append(u.lower())
    for f in FREQUENCIES:
        lines.append(f.lower())
        
    # Add actual training labels for real bigram statistics
    from config import PROCESSED_DIR
    manifest = os.path.join(PROCESSED_DIR, "manifest_clean.csv")
    
    if os.path.exists(manifest):
        df = pd.read_csv(manifest)
        for text in df["text_label"].dropna():
            if isinstance(text, str):
                lines.append(text.strip().lower())
        print(f"Added {len(df)} lines from manifest_clean.csv for LM training.")
    else:
        print(f"⚠ Warning: {manifest} not found, LM will only have lexicon words.")

    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ── 2. Run lmplz or warn if missing ──
    lmplz_bin = shutil.which("lmplz")
    if not lmplz_bin:
        # Sometimes installed inside virtualenv bin but not resolving if path weird
        # Check standard locations:
        possible = [
            os.path.join(os.path.dirname(sys.executable), "lmplz"),
            "lmplz"
        ]
        for p in possible:
            if shutil.which(p) or os.path.exists(p):
                lmplz_bin = p
                break

    if not lmplz_bin:
        print("⚠ lmplz not found in PATH.")
        print("  KenLM was installed via pip, which doesn't compile the lmplz binary.")
        print("  To get real statistics, compile KenLM from source (cmake & make) and put lmplz in PATH.")
        print("  Falling back to naive uniform ARPA generation...")
        build_arpa_programmatic(output_path)
        return

    print(f"Running {lmplz_bin} to generate 2-gram ARPA...")
    try:
        with open(corpus_path, "rb") as fin, open(output_path, "wb") as fout:
            subprocess.run([lmplz_bin, "-o", "2", "--text", "/dev/stdin", "--arpa", "/dev/stdout"],
                           stdin=fin, stdout=fout, check=False) # check=False because lmplz could return non-zero informally
        with open(output_path, "r") as f:
            header = f.readline()
        if "\\data\\" not in header:
             print("lmplz failed to produce valid ARPA.")
        else:
             print(f"  ✓ ARPA language model written to {output_path}")
    except Exception as e:
        print(f"lmplz execution failed: {e}")
        print("Falling back to naive uniform ARPA generation...")
        build_arpa_programmatic(output_path)


def build_arpa_programmatic(output_path: str):
    """
    Generate a bigram ARPA language model using uniform stats.
    Fallback when lmplz is unavailable.
    """
    # Collect all vocabulary words
    words = set()
    drug_words = set()
    for name in DRUG_NAMES:
        for w in name.split():
            words.add(w.lower())
            drug_words.add(w.lower())
    unit_words = set()
    for u in UNITS:
        words.add(u.lower())
        unit_words.add(u.lower())
    freq_words = set()
    for f in FREQUENCIES:
        for w in f.split():
            words.add(w.lower())
            freq_words.add(w.lower())

    # Add digits and common dosage numbers
    number_words = set()
    for i in range(10):
        words.add(str(i))
        number_words.add(str(i))
    for n in ["10", "20", "25", "50", "100", "150", "200", "250", "500", "650", "1000"]:
        words.add(n)
        number_words.add(n)

    words = sorted(words)
    n_unigrams = len(words) + 3  # +3 for <unk>, <s>, </s>

    # -- Unigram probabilities --
    base_logprob = round(math.log10(1.0 / n_unigrams), 4)
    backoff = -0.5  # backoff weight for unseen bigrams

    # -- Build bigrams --
    # drug → number, drug → unit, number → unit, unit → frequency
    bigrams = []
    bigram_logprob = round(math.log10(1.0 / 50), 4)  # reasonable bigram prob

    # Sample representative bigrams (not all combinations — keeps file small)
    sample_drugs = sorted(drug_words)[:40]      # top 40 drug words
    sample_numbers = sorted(number_words)[:8]    # common numbers
    sample_units = sorted(unit_words)[:8]        # common units
    sample_freqs = sorted(freq_words)[:10]       # common frequency words

    for d in sample_drugs:
        for n in sample_numbers:
            bigrams.append((d, n))
        for u in sample_units:
            bigrams.append((d, u))

    for n in sample_numbers:
        for u in sample_units:
            bigrams.append((n, u))

    for u in sample_units:
        for f in sample_freqs:
            bigrams.append((u, f))

    # <s> → drug word (sentence start with drug name)
    for d in sample_drugs:
        bigrams.append(("<s>", d))

    # any word → </s> (sentence end)
    for w in sample_units + sample_freqs:
        bigrams.append((w, "</s>"))

    n_bigrams = len(bigrams)

    # -- Write ARPA --
    lines = []
    lines.append("\\data\\")
    lines.append(f"ngram 1={n_unigrams}")
    lines.append(f"ngram 2={n_bigrams}")
    lines.append("")

    # Unigrams
    lines.append("\\1-grams:")
    lines.append(f"{-99.0:.4f}\t<unk>")
    lines.append(f"{-99.0:.4f}\t<s>\t{backoff}")
    lines.append(f"{base_logprob}\t</s>")

    for w in words:
        lines.append(f"{base_logprob}\t{w}\t{backoff}")

    lines.append("")

    # Bigrams
    lines.append("\\2-grams:")
    for w1, w2 in bigrams:
        lines.append(f"{bigram_logprob}\t{w1}\t{w2}")

    lines.append("")
    lines.append("\\end\\")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓ ARPA language model written to {output_path}")
    print(f"    Unigrams: {n_unigrams}  |  Bigrams: {n_bigrams}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "medical_lm.arpa")
    build_arpa(out)
